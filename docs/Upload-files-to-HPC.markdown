# Upload files to HPC

First clone this repository on HPC. For this you should most likely generate a
ssh key (in what ever is the latest, bestest format) on HPC and then upload it
on github. 

Compute nodes on HPC environments (like Narval) maybe air-gapped and therefore
lack internet access. Because specialized tools like huggingface-cli might not
be installed on your HPC cluster, you should upload the weights and what not
locally on the node.

Run these inside the repository on your own machine.

```bash
# The wrapper file for the training and testing data.
rsync -avzP ./metadata.csv your_username@narval.computecanada.ca:~/whisper-test/
# Replace [whisper_model_directory] with the name of the model you are going to train.
rsync -avzP ./[whisper_model_directory] your_username@narval.computecanada.ca:~/whisper-test/
# The data
rsync -avzP ./[data_dir] your_username@narval.computecanada.ca:~/whisper-test/
```

For example this is how the first dataset and model were uploaded (minus username, because that would be telling).
```bash
rsync -avzP ./metadata.csv your_username@narval.computecanada.ca:~/whisper-test/
rsync -avzP ./local-whisper-small your_username@narval.computecanada.ca:~/whisper-test/
rsync -avzP ./sand-psalm your_username@narval.computecanada.ca:~/whisper-test/
```
