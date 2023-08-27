
import torch
import time
from dataset import BigramPlusDataset

print("Script started...")


def collate_fn(batch):
  b0, b1, feat, user, target, user_target = zip(*batch)

  b0 = [torch.tensor(b) for b in b0]
  b1 = [torch.tensor(b) for b in b1]
  feat = [torch.tensor(f) for f in feat]

  b0_padded = torch.nn.utils.rnn.pad_sequence(b0,
                                              batch_first=True,
                                              padding_value=0)
  b1_padded = torch.nn.utils.rnn.pad_sequence(b1,
                                              batch_first=True,
                                              padding_value=0)
  feat_padded = torch.nn.utils.rnn.pad_sequence(feat,
                                                batch_first=True,
                                                padding_value=0)
  target_padded = torch.nn.utils.rnn.pad_sequence(target,
                                                  batch_first=True,
                                                  padding_value=-1)
  target_padded = target_padded.long()

  user = torch.tensor(user)
  user_target = torch.tensor(user_target)

  # attention mask
  attn_mask = torch.nn.Transformer.generate_square_subsequent_mask(
    target_padded.size(1) + 1)

    # zero out the first row (= user)
  attn_mask[0, :] = 0

  mask = target_padded == -1

  # add user mask to the batch
  mask = torch.cat((torch.zeros((mask.shape[0], 1)), mask), dim=1)

  return b0_padded, b1_padded, feat_padded, mask, user, attn_mask, user_target, target_padded

print("Loading dataset...")
bi_dataset = BigramPlusDataset("./Keystrokes_mobile/pretrain/train/",
                                50,
                                [1, 6, 3],
                                10)

print("Dataset loaded.")
for num_work in range(0, 11):
  print("num_workers = {}".format(num_work))
  bi_dataloader = torch.utils.data.DataLoader(bi_dataset,
                                              batch_size=512,
                                              shuffle=True,
                                              num_workers=num_work,
                                              collate_fn=collate_fn)
  
  # test time
  print("Iterating through the dataloader 100 times...")
  start = time.time()
  for i, batch in enumerate(bi_dataloader):
    time.sleep(0.5)
    if i == 100:
      break
  end = time.time()
  print("Time taken: {}".format(end - start))

print("Script ended.")
