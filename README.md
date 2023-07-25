<!-- ### Hi there 👋 -->

<!-- ### 🔭 We're currently working hard on cleaning the code. -->

<!--
**SEC4SR/SEC4SR** is a ✨ _special_ ✨ repository because its `README.md` (this file) appears on your GitHub profile.

Here are some ideas to get you started:

- 🔭 I’m currently working on ...
- 🌱 I’m currently learning ...
- 👯 I’m looking to collaborate on ...
- 🤔 I’m looking for help with ...
- 💬 Ask me about ...
- 📫 How to reach me: ...
- 😄 Pronouns: ...
- ⚡ Fun fact: ...
-->

# SEC4SR
This repository contains the code for SEC4SR (SECurity for Speaker Recogntion), a Pytorch library for adversarial machine learning research on speaker recognition. It is released in the paper [SEC4SR: A Security Analysis Platform for Speaker Recognition, Guangke Chen, Zhe Zhao, Fu Song, Sen Chen, Lingling Fan, Yang Liu.](https://arxiv.org/abs/2109.01766)

Main website: [SEC4SR website](https://sec4sr.github.io/)

<!-- Paper: Anonymous Submission to [Usenix Security 2022](https://www.usenix.org/conference/usenixsecurity22/) -->
<!-- Paper: Anonymous Submission to a conference (Under Review Now) -->
Paper: [SEC4SR: A Security Analysis Platform for Speaker Recognition](https://arxiv.org/abs/2109.01766)

Feel free to use SEC4SR for academic purpose 😄. For commercial purpose, please contact us 📫.

# 1. Usage
## 1.1 Requirements
pytorch=1.6.0, torchaudio=0.6.0, numpy=1.19.2, scipy=1.4.1, libkmcuda=6.2.3, torch-lfilter=0.0.3, pesq=0.0.2, pystoi=0.3.3, librosa=0.8.0

## 1.2 Dataset Preparation
We provide five datasets, namely, Spk10_enroll, Spk10_test, Spk10_imposter, Spk251_train and Spk_251_test. They cover all the recognition tasks (i.e., CSI-E, CSI-NE, SV and OSI). The code in `./dataset/Dataset.py` will download them automatically when they are used. You can also manually download them using the follwing links:

[Spk10_enroll, 18MB, MD5:0e90fb00b69989c0dde252a585cead85](https://drive.google.com/uc?id=1BBAo64JOahk0F3yBAovnRLZ1NvjwBy7y&export\=download)

[Spk10_test, 114MB, MD5:b0f8eb0db3d2eca567810151acf13f16](https://drive.google.com/uc?id=1WctqJtP5Es74-U7y3cFXqfHi7JkDz6g5&export\=download)

[Spk10_imposter, 212MB, MD5:42abd80e27b78983a13b74e44a67be65](https://drive.google.com/uc?id=1f1GULs0aj_Xrw8JRxe6zzvTN3r2nnOf6&export\=download)

[Spk251_train, 10GB, MD5:02bee7caf460072a6fc22e3666ac2187](https://drive.google.com/uc?id=1iGcMPiPMzcCLI7xKJLwH1L0Ff_95-tmB&export\=download)

[Spk251_test, 1GB, MD5:182dd6b17f8bcfed7a998e1597828ed6](https://drive.google.com/uc?id=1rsXzuEyi5Zqd1XAsr1_Op7mC7hqY0tsp&export\=download)

After downloading, untar them inside `./data` directory.

## 1.3 Model Preparation
### 1.3.1 Speaker Enroll (CSI-E/SV/OSI tasks)
- Download [iv_system, MD5:bfe90ec7782b54dc295e72b5bf789377](https://drive.google.com/uc?id=13yDZvM6a7W1Str2KEI7Vrm2xSdxWe7Vv&export\=download) and [xv_system, MD5:37cb3e7ca48c0da3ae72a35195aacf58](https://drive.google.com/uc?id=1HbpR6cUuPzDQLVvQTFUpIAflEa1eP-XF&export\=download), and untar them inside the reposity directory (i.e., `./`). Iv_system and xv_system contain the pre-trained ivector-PLDA and xvector-PLDA background models.
- Run `python enroll.py iv` and `python enroll.py xv` to enroll the speakers in Spk10_enroll for ivector and xvector systems. Multiple speaker models for CSI-E and OSI tasks are stored as `speaker_model_iv` and `speaker_model_xv` inside `./model_file`. Single speaker models for SV task are  stored as `speaker_model_iv_{ID}` and `speaker_model_xv_{ID}` inside `./model_file`.
- Run `python set_threshold.py -task SV iv`, `python set_threshold.py -task OSI iv`, `python set_threshold.py -task SV xv` and `python set_threshold.py -task OSI xv` to set the threshold of the system.

### 1.3.2 Natural Training (CSI-NE task)
- Sole natural training: 

  `python natural_train.py -num_epoches 30 -batch_size 128 -model_ckpt ./model_file/natural-audionet -log ./model_file/natural-audionet-log`
- Natural training with QT (q=512)

  `python adver_train.py -attacker PGD -epsilon 0.002 -max_iter 10 -defense QT -defense_param 512 -EOT_size 1 -EOT_batch_size 1 -model_ckpt ./model_file/QT-512-pgd-adver-audionet -log ./model_file/QT-512-pgd-adver-audionet-log`

### 1.3.3 Adversarial Training (CSI-NE task)
- Sole FGSM adversarial training:
  
  `python adver_train.py -attacker FGSM -epsilon 0.002 -model_ckpt ./model_file/fgsm-adver-audionet -log ./model_file/fgsm-adver-audionet-log`
- Sole PGD adversarial training:

  `python adver_train.py -attacker PGD -epsilon 0.002 -max_iter 10 -model_ckpt ./model_file/pgd-adver-audionet -log ./model_file/pgd-adver-audionet-log`
- Combining adversarial training with input transformation AT (randomized, should use EOT during training)
  
  `python adver_train.py -attacker PGD -epsilon 0.002 -max_iter 10 -defense AT -defense_param 16 -EOT_size 10 -EOT_batch_size 5 -model_ckpt ./model_file/AT-pgd-adver-audionet -log ./model_file/AT-pgd-adver-audionet-log` 

## 1.4 Generate Adversarial Examples
- Example 1: FAKEBOB attack on naturally-trained audionet model  

  `python attackMain.py -system_type audionet -model_file ./model_file/QT-512-natural-audionet -task CSI -root ./data -name Spk251_test -des ./adver-audio/QT-512-audionet-fakebob FAKEBOB -epsilon 0.002`
- Example 2: FGSM targeted attack on FC-defended ivector-plda model for OSI task. FC is randomized, using EOT

  `python attackMain.py -system_type iv -model_file ./model_file/speaker_model_iv -threshold 2.51 -task OSI -defense FC -defense_param kmeans raw 0.2 L2 -root ./data -name Spk10_imposter -des ./adver-audio/iv-fgsm -EOT_size 6 -EOT_batch_size 2 -targeted FGSM -epsilon 0.002`

## 1.5 Evaluate Adversarial Examples
- Example 1: Testing for unadaptive attack

  `python test_attack.py -system_type audionet -model_file ./model_file/QT-512-natural-audionet -root ./adver-audio -name QT-512-audionet-fakebob -defense QT -defense_param 512`
- Example 2: Testing for adaptive attack

  `python test_attack.py -system_type iv -model_file ./model_file/speaker_model_iv -threshold 2.51 -defense FC -defense_param kmeans raw 0.2 L2 -root ./adver-audio -name iv-fgsm`

In Example 1, the adversarial examples are generated on undefended audionet model, but tested on QT-defended audionet model, so it is **non-adaptive** attack.

In Example 2, the adversarial examples are generated on FC-defended iv-plda model using EOT (to overcome the randomness of FC), and also tested on FC-defended iv-plda model, so it is **adaptive** attack.

By default, targeted attack randomly selects the targeted label. If you want to control the targeted label, you can run `specify_target_label.py` and input the generated target label file to `attackMain.py` and `test_attack.py`.

`test_attack.py` can also be used to test the benign accuracy of systems. Just let `-root` and `-name` point to the benign dataset.

# 2. Extension
## MC (Model Component)
MC contains three state-of-the-art embedding-based speaker recognition models, i.e., ivector-PLDA, xvector-PLDA and AudioNet. Xvector-PLDA and AudioNet are based on neural networks while ivector-PLDA on statistic model (i.e Gaussian Mixture Model).

The flexibility and extensibility of SEC4SR make it easy to add new models. 
<!-- Just wrap the model as `torch.nn.Module` and implement `forward`, `score` and `make_decision` methods. -->
To add a new model, one can define a new subclass of the `torch.nn.Module` class and implement three methods: `forward`, `score`, and `make_decision` , then it can be evaluated using different attacks.

## DAC (Dataset Component)
We provide five datasets, namely, Spk10_enroll, Spk10_test, Spk10_imposter, Spk251_train and Spk_251_test. They cover all the recognition tasks (i.e., CSI-E, CSI-NE, SV and OSI). 

<!-- To add new datasets, one just need to define a class inheriting from `torch.utils.data.Dataset`, just like `dataset/Dataset.py`. -->
All our datasets are subclasses of the class `torch.utils.data.Dataset`. Hence, to add a new dataset, one just need to define a new subclass of `torch.utils.data.Dataset` and implement two methods: `__len__` and `__getitem__`, which defines the length and loading sequence of the dataset.

## AC (Attack Component)
SEC4SR currently incorporate four white-box attacks (FGSM, PGD, CW$_\infty$ and CW$_2$) and two black-box attacks (FAKEBOB and SirenAttack). 
<!-- To incorporate new attack algorithms, one just need to inhert from the class in `attack/Attack.py` and implement the abstract method `attack`. -->
To add a new attack, one can define a new subclass of the abstract class `Attack` and implement the `attack` method. This design ensures that the `attack` methods in different concrete `Attack` classes have the same method signature, i.e., unified API.

## DEC (Defense Component)
To secure SRSs from adversarial attack, SEC4SR provides 2 robust training methods (FGSM and PGD adversarial training) and 22 speech/speaker-dedicated input transformation methods, including our feature-level approach FEATURE COMPRESSION (FC). 
<!-- All input transformation methods are implemented as standalone python functions, making it easy to extend new methods. -->
Since all our defenses are standalone functions, adding a new defense is straightforward, one just needs to implement it as a python function accepting the input audios or features as one of its arguments.

## ADAC (Adaptive Attack Component)
All these adaptive attack techniques are implemented as standalone wrappers so that they can be easily plugged into attacks to mount adaptive attacks.
