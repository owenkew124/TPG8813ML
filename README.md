## Reusing PagPassGPT for guessing on our dataset of 51k. Paper: https://arxiv.org/abs/2404.04886 Github: https://github.com/Suxyuuu/PagPassGPT

**1.** run preprocess.py modify data_path,  processed_data_path, train_path, test_path, complete_training_dataset, PCFG_rate_file, and vocab for the locations on your file system.

**2.** Generate the vocab file within PagPass, default was a password with length 12. I set it to 20 to avoid losing a fair percentage of passwords in the training file. Run generate_vocab_file.py

**3.** Run PagPass train.py. Pass --dataset_path input as an argument which was complete_training_dataset generated in step 1.

**4.** With a model file output in PagPassGPT/models. You can then use run_guessing_for_all_users.py make sure to change the paths in the script to model_path, vocabfile_path, the target_guessing script, and original dataset text file. Also change passwords_path to where you want to output the generated passwords.
The run_guessing_for_all script runs targetted guessing for each individual user/passwords individually as seperate processes. I had a rough time getting through the entire dataset in one process with memory violation access errors being thrown at random times somewhere in transformers. 
So I ended up just having a bunch of subprocess calls and if one fails it fails. At least it let me run the retrained PagPass on the majority of users.

---
Results arent **great** we were hoping to combine the results with Owen's method but didnt get around to ironing that out.

David Oygenblik, Owen Kew
  
