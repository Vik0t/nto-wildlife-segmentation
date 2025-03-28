import torch
import cv2
import os
import numpy as np
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

class fine_tuned_sam():
    def __init__(self, model_cfg, sam2_checkpoint, scheduler, accumulation_steps):
        self.model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")
        self.predictor = SAM2ImagePredictor(self.model)
        self.accumulation_steps = accumulation_steps
        self.scheduler = scheduler
        self.scaler = torch.amp.GradScaler("cuda")
        self.train_f1_scores = []
        self.val_f1_scores = []
        self.train_losses = []
        self.val_losses = []
        self.best_f1 = 0
        self.train_mean_iou = 0
        self.valid_mean_iou = 0


    def read_batch(self, data, data_path):
         # Select a random entry
        ent = data['annotations'][np.random.randint(len(data['annotations']))]

        ind = 0
        for i in range(len(data['images'])):
           if data['images'][i]['id'] == ent['image_id']:
              ind=i
              break
        image_path = data['images'][ind]['file_name']
        image_path = os.path.join(data_path, image_path)
        Img = cv2.imread(image_path)[..., ::-1]  # Convert BGR to RGB
        binary_mask = np.zeros((693, 1344), dtype=np.uint8)
        for pts in ent['segmentation']:
          # Шаг 1: Разделить на пары (x, y)
          points = [[pts[i], pts[i + 1]] for i in range(0, len(pts), 2)]

          # Шаг 2: Обернуть в список (даже если один полигон)
          polygon = [points]

          # Шаг 3: Преобразовать в NumPy массив с типом int32
          pts_array = np.array(polygon, dtype=np.int32)
          cv2.fillPoly(binary_mask, pts_array , color=1)

        # Erode the combined binary mask to avoid boundary points
        eroded_mask = cv2.erode(binary_mask, np.ones((5, 5), np.uint8), iterations=1)

        points = []
        # Get all coordinates inside the eroded mask and choose random points
        coords = np.argwhere(eroded_mask > 0)
        if len(coords) > 0:
            yx = coords[np.random.randint(len(coords))]  # Randomly select a point
            points.append([yx[1], yx[0]])  # Append in [x, y] format (col, row)

        points = np.array(points)
        binary_mask = np.expand_dims(binary_mask, axis=-1)  # Now shape is (1024, 1024, 1)
        binary_mask = binary_mask.transpose((2, 0, 1))
        points = np.expand_dims(points, axis=1)

        # Return the image, binarized mask, points, and number of masks
        return Img, binary_mask, points, 1

    def train_1_iter(self, optimizer,train_data, step, mean_iou):    
        with torch.amp.autocast(device_type='cuda'):
            image, mask, input_point, num_masks = self.read_batch(train_data, '/home/vik0t/hackaton/segment-anything-2/data1/train/images')
            if image is None or mask is None or num_masks == 0:
                return 0
            input_label = np.ones((num_masks, 1))
            if not isinstance(input_point, np.ndarray) or not isinstance(input_label, np.ndarray):
                return 0
            if input_point.size == 0 or input_label.size == 0:
                return 0
            image = image.copy()
            self.predictor.set_image(image)
            mask_input, unnorm_coords, labels, unnorm_box = self.predictor._prep_prompts(
                input_point, input_label, box=None, mask_logits=None, normalize_coords=True
            )

            if unnorm_coords is None or labels is None or unnorm_coords.shape[0] == 0 or labels.shape[0] == 0:
                return
    
            sparse_embeddings, dense_embeddings = self.predictor.model.sam_prompt_encoder(
                points=(unnorm_coords, labels), boxes=None, masks=None
            )
    
            batched_mode = unnorm_coords.shape[0] > 1
            high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in self.predictor._features["high_res_feats"]]
    
            low_res_masks, prd_scores, _, _ = self.predictor.model.sam_mask_decoder(
                image_embeddings=self.predictor._features["image_embed"][-1].unsqueeze(0),
                image_pe=self.predictor.model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True,
                repeat_image=batched_mode,
                high_res_features=high_res_features,
            )
    
            prd_masks = self.predictor._transforms.postprocess_masks(low_res_masks, self.predictor._orig_hw[-1])

            gt_mask = torch.tensor(mask.astype(np.float32)).cuda()
            prd_mask = torch.sigmoid(prd_masks[:, 0])

            seg_loss = (-gt_mask * torch.log(prd_mask + 1e-6) - (1 - gt_mask) * torch.log((1 - prd_mask) + 1e-6)).mean()

            inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
            iou = inter / (gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter)

            score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
            loss = seg_loss + score_loss * 0.05
            loss = loss / self.accumulation_steps


            self.scaler.scale(loss).backward()

            torch.nn.utils.clip_grad_norm_(self.predictor.model.parameters(), max_norm=1.0)

            self.scheduler.step() 

            if step % self.accumulation_steps == 0:
                self.scaler.step(optimizer)
                self.scaler.update()
                self.predictor.model.zero_grad()
    


            mean_iou = mean_iou * 0.99 + 0.01 * np.mean(iou.cpu().detach().numpy())
            self.train_f1_scores.append(mean_iou) 
            self.train_losses.append(seg_loss.detach().cpu().numpy())

            if step % 100 == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                print(f"Step {step}: Current LR = {current_lr:.8f}, IoU = {mean_iou:.6f}, Seg Loss = {seg_loss:.6f}")
        return mean_iou
    
    def validate(self, optimizer, val_data, step, mean_iou):
        self.predictor.model.eval()
        with torch.amp.autocast(device_type='cuda'):
            with torch.no_grad():
                image, mask, input_point, num_masks = self.read_batch(val_data, '/home/vik0t/hackaton/segment-anything-2/data1/val/images')
                if image is None or mask is None or num_masks == 0:
                    return

                input_label = np.ones((num_masks, 1))
                if not isinstance(input_point, np.ndarray) or not isinstance(input_label, np.ndarray):
                    return

                if input_point.size == 0 or input_label.size == 0:
                    return
                image = image.copy()
                self.predictor.set_image(image)
                mask_input, unnorm_coords, labels, unnorm_box = self.predictor._prep_prompts(
                    input_point, input_label, box=None, mask_logits=None, normalize_coords=True
                )

                if unnorm_coords is None or labels is None or unnorm_coords.shape[0] == 0 or labels.shape[0] == 0:
                    return

                sparse_embeddings, dense_embeddings = self.predictor.model.sam_prompt_encoder(
                    points=(unnorm_coords, labels), boxes=None, masks=None
                )
    
                batched_mode = unnorm_coords.shape[0] > 1
                high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in self.predictor._features["high_res_feats"]]
                low_res_masks, prd_scores, _, _ = self.predictor.model.sam_mask_decoder(
                    image_embeddings=self.predictor._features["image_embed"][-1].unsqueeze(0),
                    image_pe=self.predictor.model.sam_prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=True,
                    repeat_image=batched_mode,
                    high_res_features=high_res_features,
                )
    
                prd_masks = self.predictor._transforms.postprocess_masks(low_res_masks, self.predictor._orig_hw[-1])
    
                gt_mask = torch.tensor(mask.astype(np.float32)).cuda()
                prd_mask = torch.sigmoid(prd_masks[:, 0])
    
                seg_loss = (-gt_mask * torch.log(prd_mask + 1e-6)
                            - (1 - gt_mask) * torch.log((1 - prd_mask) + 1e-6)).mean()
    
                inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
                iou = inter / (gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter)
    
                score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
                loss = seg_loss + score_loss * 0.05
                loss = loss / self.accumulation_steps

                if step % 1000 == 0:
                    FINE_TUNED_MODEL = self.FINE_TUNED_MODEL_NAME + "_" + str(step) + ".pt"
                    torch.save(self.predictor.model.state_dict(), FINE_TUNED_MODEL)

                mean_iou = mean_iou * 0.99 + 0.01 * np.mean(iou.cpu().detach().numpy())
                self.val_f1_scores.append(mean_iou)
                self.val_losses.append(seg_loss.detach().cpu().numpy())

                if step > 300 and mean_iou > self.best_f1:
                    FINE_TUNED_MODEL = "BEST" + ".pt"
                    torch.save(self.predictor.model.state_dict(), FINE_TUNED_MODEL)
                    self.best_f1 = mean_iou

                if step % 100 == 0:
                    current_lr = optimizer.param_groups[0]["lr"]
                    print(f"Step {step}: Current LR = {current_lr:.8f}, Valid_IoU = {mean_iou:.6f}, Valid_Seg Loss = {seg_loss:.6f}")
        return mean_iou

    def train(self, NO_OF_STEPS, optimizer, train_data, val_data):
        self.train_mean_iou = 0
        self.valid_mean_iou = 0
 
        for step in range(1, NO_OF_STEPS + 1):
            self.train_mean_iou = self.train_1_iter(optimizer,train_data, step, self.train_mean_iou)
            self.valid_mean_iou = self.validate(optimizer,val_data, step, self.valid_mean_iou)
