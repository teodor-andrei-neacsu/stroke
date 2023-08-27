## StrokeNet

Learning key and user embeddings for keystroke biometric authentication.

DO LIST:
- DATA: assure that there is no data leakage between the training and testing subjects !!! (save what files were used for training and testing)
- Modify the main to properly use hydra 
- From the raw data generate 2 datasets -> training subjects where the ratio between train and test is whatever (12/3)
                                        -> testing subjects where the ratio is (1, 3, 5, 7, 10/ 5)
- Pretrain the model on the training subjects
- Evaluate the model on the training subjects using the unseen sequences
- Add checkpointing to the best model
- Add the loading of the best model from the checkpoint
- Modify the model in order to use another embedding table for the users that will be initilized from the avg of the embeddings of the training subjects
- Evaluate the model on the testing subjects using the unseen sequences (using all the configurations)
