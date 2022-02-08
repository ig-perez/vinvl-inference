# General
from abc import *

# Encoder
from numpy.core.fromnumeric import transpose

from fellow.config_maskrcnn_benchmark import get_cfg_defaults
from fellow.config_scene_graph_benchmark import get_sg_cfg_defaults
from fellow.frcnn import AttrRCNN
from fellow.checkpointer import DetectronCheckpointer


import numpy as np


# Decoder
from oscar.modeling.modeling_bert import BertForImageCaptioning
from transformers.pytorch_transformers import BertTokenizer, BertConfig
from oscar.run_captioning import CaptionTensorizer
import numpy as np
import torch



# Inference
import cv2
import matplotlib.pyplot as plt


class BaseEncoder(object, metaclass = ABCMeta):
    @abstractmethod
    def encode(self, imgs):
        pass

    def __call__(self, imgs):
        return self.encode(imgs)

class Encoder(BaseEncoder):
    def __init__(self,
                 config_file="fellow/vinvl_x152c4.yaml",
                 opts=["TEST.IMS_PER_BATCH", "2", "MODEL.WEIGHT", "fellow/od_models/vinvl_vg_x152c4.pth",
                       "MODEL.ROI_HEADS.NMS_FILTER", "1", "MODEL.ROI_HEADS.SCORE_THRESH", "0.2",
                       "TEST.IGNORE_BOX_REGRESSION", "True", "MODEL.ATTRIBUTE_ON", "True", "TEST.OUTPUT_FEATURE",
                       "True"],
                 ckpt="fellow/od_models/vinvl_vg_x152c4.pth",
                 DEVICE="cuda",
                 MIN_BOXES=10,
                 MAX_BOXES=100,
                 conf_threshold=0.2):

        cfg = get_cfg_defaults()
        sg_cfg = get_sg_cfg_defaults()

        cfg.set_new_allowed(True)
        cfg.merge_from_other_cfg(sg_cfg)
        cfg.set_new_allowed(False)
        cfg.merge_from_file(config_file)
        cfg.local_rank = 0
        cfg.merge_from_list(opts)

        self.model = AttrRCNN(cfg)
        self.model.to(DEVICE)

        checkpointer = DetectronCheckpointer(cfg, self.model)
        _ = checkpointer.load(ckpt, use_latest=True)

        self.DEVICE = DEVICE

        self.MIN_BOXES = MIN_BOXES
        self.MAX_BOXES = MAX_BOXES

        self.conf_threshold = conf_threshold

    def permute_img(self, img):
        img = torch.Tensor(img).to(self.DEVICE)
        img = img.permute((2, 0, 1))
        return img

    def compute_on_image(self, images):
        self.model.eval()
        with torch.no_grad():
            output = self.model(images, None)
        output = [o.to('cpu') for o in output]
        return output

    def encode(self, imgs):

        for i in range(len(imgs)):
            imgs[i] = self.permute_img(imgs[i])
        pred = self.compute_on_image(imgs)

        outs = []
        for img, p in zip(imgs, pred):

            features = p.get_field('box_features')
            boxes = []
            for bs in p.get_field('boxes_all'):
                boxes.append(bs[0].numpy())
            boxes = np.array(boxes)
            conf = p.get_field('scores')
            labels = p.get_field('labels')

            keep_boxes = np.where(conf >= self.conf_threshold)[0]

            boxes = boxes[keep_boxes]
            features = features[keep_boxes]
            labels = labels[keep_boxes]

            img_h = img.shape[1]
            img_w = img.shape[2]

            box_width = boxes[:, 2] - boxes[:, 0]
            box_height = boxes[:, 3] - boxes[:, 1]

            scaled_width = box_width / img_w
            scaled_height = box_height / img_h
            scaled_x = boxes[:, 0] / img_w
            scaled_y = boxes[:, 1] / img_h

            scaled_width = scaled_width[..., np.newaxis]
            scaled_height = scaled_height[..., np.newaxis]
            scaled_x = scaled_x[..., np.newaxis]
            scaled_y = scaled_y[..., np.newaxis]

            spatial_feat = np.concatenate((scaled_x, scaled_y, scaled_x + scaled_width, scaled_y + scaled_height,
                                           scaled_width, scaled_height), axis=1)
            full_feat = np.concatenate((features, spatial_feat), axis=1)
            outs.append([full_feat, labels])

        return outs

class BaseDecoder(object, metaclass=ABCMeta):
    """
    Base Class to decode Encoded Information.
    """
    @abstractmethod
    def decode(self, args):
        pass

    def __call__(self, args):
	    return self.decode(args)

class Decoder(BaseDecoder):
    """
    Captioning Decoder Class to decode Encoded Information

    """

    def __init__(self, checkpoint, device='cuda'):
        """
        Initialized the Captioning Decoder
        : param checkpoint (string) : Location of Checkpoint
        : param device     (string) : Device to load the Decoder

        : return : None
        """
        self.checkpoint = checkpoint

        self.config = BertConfig.from_pretrained(self.checkpoint)
        self.config.output_hidden_states = True

        self.tokenizer = BertTokenizer.from_pretrained(self.checkpoint)
        self.model = BertForImageCaptioning.from_pretrained(self.checkpoint, config=self.config)

        self.device = device
        self.model.to(self.device)
        self.model.eval()

        self.tensorizer = CaptionTensorizer(self.tokenizer, max_img_seq_length=50, max_seq_length=70, \
                                            max_seq_a_length=40, mask_prob=0.15, max_masked_tokens=3, is_train=False)

    def decode(self, args):
        """
        Main function to decode encoded information to image caption
        : param args      (list) :  list of (feature vector, object labels)

        : return captions (list) : list of image captions
        """
        examples = [[], [], [], [], []]
        caption = ""
        for i in range(len(args)):
            od_labels = " ".join([str(t) for t in np.array(args[i][1])])
            example = self.tensorizer.tensorize_example(caption, torch.Tensor(args[i][0]), text_b=od_labels)
            cls_token_id, sep_token_id, pad_token_id, mask_token_id, period_token_id = \
                self.tokenizer.convert_tokens_to_ids([self.tokenizer.cls_token, self.tokenizer.sep_token, \
                                                      self.tokenizer.pad_token, self.tokenizer.mask_token, '.'])
            for j in range(5):
                examples[j].append(example[j])

        examples = tuple(torch.stack(ex) for ex in examples)
        with torch.no_grad():
            batch = tuple(t.to(self.device) for t in examples)
            inputs = {
                'input_ids': batch[0], 'attention_mask': batch[1],
                'token_type_ids': batch[2], 'img_feats': batch[3],
                'masked_pos': batch[4],
            }
            inputs_param = {'is_decode': True,
                            'do_sample': False,
                            'bos_token_id': cls_token_id,
                            'pad_token_id': pad_token_id,
                            'eos_token_ids': [sep_token_id],
                            'mask_token_id': mask_token_id,
                            # for adding od labels
                            'add_od_labels': True, 'od_labels_start_posid': 40,
                            # hyperparameters of beam search
                            'max_length': 20,
                            'num_beams': 5,
                            "temperature": 1,
                            "top_k": 0,
                            "top_p": 1,
                            "repetition_penalty": 1,
                            "length_penalty": 1,
                            "num_return_sequences": 3,
                            "num_keep_best": 3,
                            }
            inputs.update(inputs_param)
            output = self.model(**inputs)

        captions = []

        for i in range(len(args)):
            captions.append(self.tokenizer.decode(output[0][i * 3][0].tolist(), skip_special_tokens=True))
        return captions

if __name__ == "__main__":
    # This is to run on a notebook
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    encoder = Encoder()
    decoder = Decoder('fellow/coco_captioning_base_scst/checkpoint-15-66405')  # base model w/ CIDEr optimization

    img1 = cv2.imread('fellow/images/woman.png', cv2.IMREAD_UNCHANGED)
    img2 = cv2.imread('fellow/images/food.png', cv2.IMREAD_UNCHANGED)
    img3 = cv2.imread('fellow/images/man.png', cv2.IMREAD_UNCHANGED)

    fig = plt.figure(figsize=(20, 12))
    for i in range(1, 4):
        fig.add_subplot(fig.add_subplot(1, 3, i))
        plt.imshow(cv2.cvtColor(eval('img' + str(i)), cv2.COLOR_BGR2RGB))
    plt.show()

    out = decoder(encoder([img1, img2, img3]))

    print(out)

