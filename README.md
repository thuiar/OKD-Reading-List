# Open Knowledge Discovery Reading List

This is an open knowledge discovery reading list maintained by THUIAR team. As real-world scenarios are usually open settings, it is crucial to discovery these open
knowledge (e.g., new user intents in dialogue system, image open set and so on) to improve the quality of machine learning systems. 

Our list is still incomplete and the taxonomy may be inappropriate. We will keep adding papers to improve the list, and welcome to **Pull Request** ! If you have any suggestions, please contact zhang-hl20@mails.tsinghua.edu.cn.

Contributors: Hanlei Zhang, Shaojie Zhao, Kang Zhao, Ting-En Lin, Rui Lyu.


## Contents
* [Natural Language Processing](#Natural_Language_Processing)
    * [Toolkit](#Toolkit)
    * [Intent Detection](#Intent_Detection)
    * [Out-of-domain Detection](#Out-of-domain_Detection)
    * [Intent Discovery](#Intent_Discovery)
 * [Computer Vision](#Computer_Vision)
    * [Out of Domain Detection](#Out_of_Domain_Detection)
    * [Open Set Recognition / Open World Classification](#Open_Set_Recognition)
    * [New Category Discovery](#New_Category_Discovery)
* [Machine Learning](#ML)
    * [Deep Clustering](#Deep_Clustering) 
<!-- * [Multimodality](#Multimodality)
    * [Dialogue System](#Dialogue_System)  -->
    
<h2 id="Natural_Language_Processing">Natural Language Processing</h2> 

<h3 id="Toolkit">Toolkit</h3> 

* Hanlei Zhang, Xiaoteng Li, Hua Xu, Panpan Zhang, Kang Zhao, Kai Gao. 2021. **TEXTOIR: An Integrated and Visualized Platform for Text Open Intent Recognition**. In *Proceedings of ACL 2021*. [[paper](https://aclanthology.org/2021.acl-demo.20.pdf)] [[toolkit](https://github.com/thuiar/TEXTOIR)] [[demo](https://github.com/thuiar/TEXTOIR-DEMO)]

<h3 id="Open_Intent_Detection">Intent Detection</h3> 

* Liming Zhan, Haowen Liang, Bo Liu, Lu Fan, Xiaoming Wu, Albert Y.S. Lam. 2021. **Out-of-Scope Intent Detection with Self-Supervision and Discriminative Training**. In *Proceedings of ACL-IJCNLP 2021*. [[paper](https://arxiv.org/pdf/2106.08616.pdf)] [[code](https://github.com/liam0949/DCLOOS)]
* Yilin Shen, Yen-Chang Hsu, Avik Ray, Hongxia Jin. 2021. **Enhancing the generalization for Intent Classification and Out-of-Domain Detection in SLU**. In *Proceedings of ACL-IJCNLP 2021*. [[paper](https://arxiv.org/pdf/2106.14464.pdf)]
* DongHyun Choi, Myeong Cheol Shin, EungGyun Kim, Dong Ryeol Shin. 2021. **OutFlip: Generating Out-of-Domain Samples for Unknown Intent Detection with Natural Language Attack**. In *Proceedings of ACL-IJCNLP 2021 Findings* .[[paper](https://arxiv.org/pdf/2105.05601.pdf)]
* Yawen Ouyang, Jiasheng Ye, Yu Chen, Xinyu Dai, Shujian Huang, Jiajun Chen. 2021. **Energy-based Unknown Intent Detection with Data Manipulation**. In *Proceedings of ACL-IJCNLP 2021 Findings* .[[paper](https://arxiv.org/pdf/2107.12542.pdf)]
* Zhiyuan Zeng, Keqing He, Yuanmeng Yan, Zijun Liu, Yanan Wu, Hong Xu, Huixing Jiang, Weiran Xu. 2021. **Modeling Discriminative Representations for Out-of-Domain Detection with Supervised Contrastive Learning**. In *Proceedings of ACL-IJCNLP 2021*. [[paper](https://arxiv.org/pdf/2105.14289.pdf)][[code](https://github.com/parZival27/supervised-contrastive-learning-for-out-of-domain-detection)]
* Hanlei Zhang, Hua Xu and Ting-En Lin. 2021. **Deep Open Intent Classification with Adaptive Decision Boundary**. In *Proceedings of AAAI 2021*. [[paper](https://arxiv.org/pdf/2012.10209.pdf)] [[code](https://github.com/thuiar/Adaptive-Decision-Boundary)]
* Guangfeng Yan, Lu Fan, Qimai Li, Han Liu, Xiaotong Zhang, Xiao-Ming Wu and Albert Y.S. Lam. 2020. **Unknown Intent Detection Using Gaussian Mixture Model with an Application to Zero-shot Intent Classification**. In *Proceedings of ACL 2020*. [[paper](https://www.aclweb.org/anthology/2020.acl-main.99.pdf)]
* Iñigo Casanueva, Tadas Temčinas, Daniela Gerz, Matthew Henderson, Ivan Vulić. 2020. **Efficient Intent Detection with Dual Sentence Encoders**. In *Proceedings of ACL 2020*. [[paper](https://www.aclweb.org/anthology/2020.nlp4convai-1.5.pdf)] [[dataset](https://github.com/PolyAI-LDN/task-specific-datasets)]
* Ting-En Lin and Hua Xu. 2019. **Deep Unknown Intent Detection with Margin Loss**. In *Proceedings of ACL 2019*. [[paper](https://www.aclweb.org/anthology/P19-1548.pdf)] [[code](https://github.com/thuiar/DeepUnkID)]
* Congying Xia, Chenwei Zhang, Xiaohui Yan, Yi Chang and Philip S. Yu. 2018. **Zero-shot User Intent Detection via Capsule Neural Networks**. In *Proceedings of EMNLP 2018*. [[paper](https://www.aclweb.org/anthology/D18-1348.pdf)] [[code](https://github.com/congyingxia/ZeroShotCapsule)]


<h3 id="Out-of-domain_Detection">Out-of-domain Detection</h3> 

* Xiaoya Li, Jiwei Li, Xiaofei Sun, Chun Fan, Tianwei Zhang, Fei Wu, Yuxian Meng, Jun Zhang. 2021. **kFolden: k-Fold Ensemble for Out-of-Distribution Detection**. In *Proceedings of EMNLP 2021*. [[paper](https://aclanthology.org/2021.emnlp-main.248/)][[code](https://github.com/ShannonAI/kfolden-ood-detection)]
* Yinhe Zheng, Guanyi Chen, and Minlie Huang. 2020. **Out-of-domain Detection for Natural Language Understanding in Dialog Systems**. *IEEE Transactions on Audio, Speech and Language Processing*. [[paper](https://ieeexplore.ieee.org/document/9052492)]
* Gangal Varun, Arora Abhinav, Einolghozati Arash and Gupta Sonal. 2020. **Likelihood Ratios and Generative Classifiers for Unsupervised Out-of-Domain Detection in Task Oriented Dialog**. In *Proceedings of AAAI 2020*. [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/6280)]
* Larson Stefan, Mahendran Anish, Peper Joseph J, Clarke Christopher, Lee Andrew, Hill Parker, Kummerfeld Jonathan K, Leach Kevin, Laurenzano Michael A., Tang Lingjia and Mars Jason. 2019. **An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction**. In *Proceedings of EMNLP-IJCNLP 2019*. [[paper](https://www.aclweb.org/anthology/D19-1131.pdf)] [[dataset](https://github.com/clinc/oos-eval)]
* Joo-Kyung Kim and Young-Bum Kim. 2018. **Joint Learning of Domain Classification and Out-of-Domain Detection with Dynamic Class Weighting for Satisficing False Acceptance Rates**.  In *Proceedings of INTERSPEECH 2018*. [[paper](https://arxiv.org/pdf/1807.00072.pdf)]
* Seonghan Ryu, Sangjun Koo, Hwanjo Yu, and Gary Geunbae Lee. 2018. **Out-of-domain Detection based on Generative Adversarial Network**.  In *Proceedings of EMNLP 2018*. [[paper](https://www.aclweb.org/anthology/D18-1077.pdf)] 
* Ian Lane, Tatsuya Kawahara, Tomoko Matsui and Satoshi Nakamura. 2006. **Out-of-Domain Utterance Detection Using Classification Confidences of Multiple Topics**. *IEEE Transactions on Audio, Speech, and Language Processing*. [[paper](https://ieeexplore.ieee.org/document/4032779)]

<h3 id="Open_World_Classification">Open World Classification</h3> 

* Lei Shu, Yassine Benajiba, Saab Mansour, Yi Zhang. **ODIST: Open World Classification via Distributionally Shifted Instances**. 2021. In *Proceedings of EMNLP 2021 Findings*. [[paper](https://aclanthology.org/2021.findings-emnlp.316.pdf)]
* Hu Xu, Bing Liu, Lei Shu and P. Yu. **Open-world Learning and Application to Product Classification**. 2019.  In *Proceedings of WWW 2019*. [[paper](https://arxiv.org/abs/1809.06004)] [[code](https://github.com/howardhsu/Meta-Open-World-Learning)]
* Lei Shu, Hu Xu and Bing Liu. 2017. **DOC: Deep Open Classification of Text Documents**. In *Proceedings of EMNLP 2017*. [[paper](https://www.aclweb.org/anthology/D17-1314.pdf)] [[code](https://github.com/leishu02/EMNLP2017_DOC)]
* Geli Fei and Bing Liu. **Breaking the Closed World Assumption in Text Classification**. 2016.  In *Proceedings of HLT-NAACL 2016*. [[paper](https://www.aclweb.org/anthology/N16-1061.pdf)] 

<h3 id="Intent_Discovery">Intent Discovery</h3> 

* Hanlei Zhang, Hua Xu, Ting-En Lin and Rui Lyu. 2021. **Discovering New Intents with Deep Aligned Clustering**. In *Proceedings of AAAI 2021*. [[paper](https://arxiv.org/pdf/2012.08987.pdf)] [[code](https://github.com/thuiar/DeepAligned-Clustering)]
* Nikhita Vedula, Nedim Lipka, Pranav Maneriker and Srinivasan Parthasarathy. **Open Intent Extraction from Natural Language Interactions**.  In *Proceedings of WWW 2020*. [[paper](https://dl.acm.org/doi/pdf/10.1145/3366423.3380268)]  
* Ting-En Lin, Hua Xu and Hanlei Zhang. 2020. **Discovering New Intents via Constrained Deep Adaptive Clustering with Cluster Refinement**.  In *Proceedings of AAAI 2020*. [[paper](https://arxiv.org/pdf/1911.08891.pdf)] [[code](https://github.com/thuiar/CDAC-plus)]
* Hugh Perkins, Yi Yang. 2019. **Dialog Intent Induction with Deep Multi-View Clustering**.  In *Proceedings of EMNLP 2019*. [[paper](https://www.aclweb.org/anthology/D19-1413.pdf)] [[code](https://github.com/asappresearch/dialog-intent-induction)]
* Iryna Haponchyk*, Antonio Uva*, Seunghak Yu, Olga Uryupina and Alessandro Moschitti. 2018. **Supervised Clustering of Questions into Intents for Dialog System Applications**. In *Proceedings of EMNLP 2018*. [[paper](https://www.aclweb.org/anthology/D18-1254.pdf)] [[dataset](https://ikernels-portal.disi.unitn.it/repository/intent-qa/)]
* Padmasundari and Srinivas Bangalore. 2018. **Intent Discovery Through Unsupervised Semantic Text Clustering**. In *Proceedings of INTERSPEECH 2018*. [[paper](https://www.isca-speech.org/archive/Interspeech_2018/pdfs/2436.pdf)]
* Toma´s Brychc ˇ ´ın and Pavel Kral´. 2017. **Unsupervised Dialogue Act Induction using Gaussian Mixtures**.  In *Proceedings of EACL 2017*. [[paper](https://www.aclweb.org/anthology/E17-2078.pdf)] 
* Dilek Hakkani-Tür, Yun-Cheng Ju, Geoff Zweig and Gokhan Tur. 2015. **Clustering Novel Intents in a Conversational Interaction System with Semantic Parsing**. In *Proceedings of INTERSPEECH 2015*. [[paper](https://www.isca-speech.org/archive/interspeech_2015/papers/i15_1854.pdf)]
* George Forman, Hila Nachlieli, and Renato Keshet. 2015. **Clustering by intent: A semi-supervised method to discover relevant clusters incrementally**.  In *Proceedings of ECML-PKDD 2015*. [[paper](https://link.springer.com/chapter/10.1007/978-3-319-23461-8_2)]
* Dilek Hakkani-Tür, Asli Celikyilmaz, Larry Heck and Gokhan Tur. 2013.  **A Weakly-Supervised Approach for Discovering New User Intents from Search Query Logs**. In *Proceedings of INTERSPEECH 2013*. [[paper](https://www.isca-speech.org/archive/archive_papers/interspeech_2013/i13_3780.pdf)]

<h2 id="Computer_Vision">Computer Vision</h2> 

<h3 id="Out_of_Domain_Detection">Out-of-domain Detection</h3> 

* Vikash Sehwag, Mung Chiang, Prateek Mittal. 2021.
**SSD: A Unified Framework for Self-Supervised Outlier Detection**. In *Proceedings of ICLR 2021*. [[paper](https://openreview.net/pdf?id=v5gjXpmR8J)] [[code](https://github.com/inspire-group/SSD)]
* Yen-Chang Hsu, Yilin Shen, Hongxia Jin, Zsolt Kira. 2020. **Generalized ODIN: Detecting Out-of-distribution Image without Learning from Out-of-distribution Data**.  In *Proceedings of CVPR 2020*. [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Hsu_Generalized_ODIN_Detecting_Out-of-Distribution_Image_Without_Learning_From_Out-of-Distribution_Data_CVPR_2020_paper.pdf)] 
* Julian Bitterwolf, Alexander Meinke, Matthias Hein. 2020. **Certifiably Adversarially Robust Detection of Out-of-Distribution Data**. In *Proceedings of NeurIPS 2020*. In *Proceedings of NeurIPS 2020*. [[paper](https://proceedings.neurips.cc/paper/2020/file/b90c46963248e6d7aab1e0f429743ca0-Paper.pdf)] [[code](https://github.com/j-cb/GOOD)]
* Taewon Jeong, Heeyoung Kim. 2020. **OOD-MAML: Meta-Learning for Few-Shot Out-of-Distribution Detection and Classification**. In *Proceedings of NeurIPS 2020*. [[paper](https://proceedings.neurips.cc/paper/2020/file/28e209b61a52482a0ae1cb9f5959c792-Paper.pdf)]
* Joan Serrà, David Álvarez, Vicenç Gómez, Olga Slizovskaia, José F. Núñez, Jordi Luque. 2020. **Input Complexity and Out-of-distribution Detection with Likelihood-based Generative Models**. In *Proceedings of ICLR 2020*. [[paper](https://openreview.net/pdf?id=SyxIWpVYvr)] 
* Eric Nalisnick, Akihiro Matsukawa, Yee Whye Teh, Dilan Gorur and Balaji Lakshminarayanan. 2019. **Do Deep Generative Models Know What They Don't Know?** In *Proceedings of ICLR 2019*. [[paper](https://openreview.net/pdf?id=H1xwNhCcYm)] [[code](https://github.com/y0ast/Glow-PyTorch)]
* Qing Yu and Kiyoharu Aizawa. 2019. **Unsupervised Out-of-Distribution Detection by Maximum Classifier Discrepancy**.  In *Proceedings of ICCV 2019*. [[paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Yu_Unsupervised_Out-of-Distribution_Detection_by_Maximum_Classifier_Discrepancy_ICCV_2019_paper.pdf)] [[code](https://github.com/Mephisto405/Unsupervised-Out-of-Distribution-Detection-by-Maximum-Classifier-Discrepancy)]
* Jie Ren, Peter J. Liu, Emily Fertig, Jasper Snoek, Ryan Poplin, Mark A. DePristo, Joshua V. Dillon and Balaji Lakshminarayanan. 2019. **Likelihood Ratios for Out-of-Distribution Detection**. In *Proceedings of NeurIPS 2019*. [[paper](https://papers.nips.cc/paper/2019/file/1e79596878b2320cac26dd792a6c51c9-Paper.pdf)] [[code](https://github.com/google-research/google-research)]
* Alireza Shafaei, Mark Schmidt and  James J. Little. 2019. **A Less Biased Evaluation of Out-of-distribution Sample Detectors**. In *Proceedings of BMVC 2019*. [[paper](https://bmvc2019.org/wp-content/uploads/papers/0333-paper.pdf)] [[code](https://github.com/ashafaei/OD-test)]
* Shiyu Liang, Yixuan Li and R. Srikant. 2018. **Enhancing The Reliability of Out-of-distribution Image Detection in Neural Networks**. In *Proceedings of ICLR 2018*. [[paper](https://openreview.net/pdf?id=H1VGkIxRZ)] [[code](https://github.com/facebookresearch/odin)]
* Dan Hendrycks and Kevin Gimpel. 2017. **A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks**. In *Proceedings of ICLR 2017*. [[paper](https://arxiv.org/pdf/1610.02136.pdf)] [[code](https://github.com/facebookresearch/odin)]
* Anh Nguyen, Jason Yosinski and Jeff Clune. 2015. **Deep Neural Networks Are Easily Fooled: High Confidence Predictions for Unrecognizable Images**. In *Proceedings of CVPR 2015*. [[paper](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Nguyen_Deep_Neural_Networks_2015_CVPR_paper.pdf)]


<h3 id="Open_Set_Recognition">Open Set Recognition / Open World Classification</h3> 

* Da-Wei Zhou, Han-Jia Ye, De-Chuan Zhan. 2021. **Learning Placeholders for Open-Set Recognition**. In *Proceedings of CVPR 2021.* [[paper](https://arxiv.org/pdf/2103.15086.pdf)]
* Hong-Ming Yang, Xu-Yao Zhang, Fei Yin, Qing Yang and Cheng-Lin Liu. 2020. **Convolutional Prototype Network for Open Set Recognition**. *IEEE Transactions on Pattern Analysis and Machine Intelligence*. [[paper](https://ieeexplore.ieee.org/abstract/document/9296325)] 
* Chuanxing Geng and Songcan Chen. 2020. **Collective decision for open set recognition**. *IEEE Transactions on Knowledge and Data Engineering*. [[paper](https://ieeexplore.ieee.org/abstract/document/9296325)] 
* Pramuditha Perera, Vlad I. Morariu, Rajiv Jain, Varun Manjunatha, Curtis Wigington, Vicente Ordonez and Vishal M. Patel. 2020. **Generative-discriminative Feature Representations for Open-set Recognition**. In *Proceedings of CVPR 2020*. [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Perera_Generative-Discriminative_Feature_Representations_for_Open-Set_Recognition_CVPR_2020_paper.pdf)] 
* Xin Sun, Zhenning Yang, Chi Zhang, Guohao Peng and Keck-Voon Ling. 2020. **Conditional Gaussian Distribution Learning for Open Set Recognition**. In *Proceedings of CVPR 2020*. [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Sun_Conditional_Gaussian_Distribution_Learning_for_Open_Set_Recognition_CVPR_2020_paper.pdf)] [[code](https://github.com/BraveGump/CGDL-for-Open-Set-Recognition)]
* Bo Liu, Hao Kang, Haoxiang Li, Gang Hua and Nuno Vasconcelos. 2020. **Few-Shot Open-Set Recognition Using Meta-Learning**. In *Proceedings of CVPR 2020*. [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Liu_Few-Shot_Open-Set_Recognition_Using_Meta-Learning_CVPR_2020_paper.pdf)]
* Guangyao Chen, Limeng Qiao, Yemin Shi, Peixi Peng, Jia Li, Tiejun Huang, Shiliang Pu and Yonghong Tian. 2020. **Learning Open Set Network with Discriminative Reciprocal Points**. In *Proceedings of ECCV 2020*. [[paper](https://link.springer.com/chapter/10.1007%2F978-3-030-58580-8_30)]
* Qing Yu, Daiki Ikami, Go Irie and Kiyoharu Aizawa. 2020. **Multi-Task Curriculum Framework for Open-Set Semi-Supervised Learning**. In *Proceedings of ECCV 2020*. [[paper](https://link.springer.com/chapter/10.1007/978-3-030-58610-2_26)]
* Chuanxing Geng, Sheng-Jun Huang and Songcan Chen. 2020. **Recent Advances in Open Set Recognition: A Survey**. In *IEEE Transactions on Pattern Analysis and Machine Intelligence*. [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9040673)] 
* T. E. Boult, S. Cruz, A.R. Dhamija, M. Gunther, J. Henrydoss and W.J. Scheirer. **Learning and the Unknown: Surveying Steps toward Open World Recognition**. In *Proceedings of AAAI 2019*. [[paper](https://www.vast.uccs.edu/~tboult/PAPERS/Learning_and_the_Unknown_Surveying_Steps_Toward_Open_World_Recognition_AAAI19.pdf)]
* Liu Ziwei, Miao Zhongqi, Zhan Xiaohang, Wang Jiayun, Gong Boqing and Yu Stella X. 2019. **Large-Scale Long-Tailed Recognition in an Open World**. In *Proceedings of CVPR 2019*. [[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_Large-Scale_Long-Tailed_Recognition_in_an_Open_World_CVPR_2019_paper.pdf)] [[code](https://github.com/zhmiao/OpenLongTailRecognition-OLTR)]
* Ryota Yoshihashi, Wen Shao, Rei Kawakami, Shaodi You, Makoto Iida and Takeshi Naemura. 2019. **Classification-Reconstruction Learning for Open-Set Recognition**. In *Proceedings of CVPR 2019*. [[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Yoshihashi_Classification-Reconstruction_Learning_for_Open-Set_Recognition_CVPR_2019_paper.pdf)] [[code](https://github.com/zhmiao/OpenLongTailRecognition-OLTR)]
* Pramuditha Perera and Vishal M. Patel. 2019. **Deep Transfer Learning for Multiple Class Novelty Detection**. In *Proceedings of CVPR 2019*. [[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Perera_Deep_Transfer_Learning_for_Multiple_Class_Novelty_Detection_CVPR_2019_paper.pdf)] [[code](https://github.com/PramuPerera/TransferLearningNovelty)]
* Poojan Oza and Vishal M. Patel. 2019. **C2AE: Class Conditioned Auto-Encoder for Open-set Recognition**. In *Proceedings of CVPR 2019*. [[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Oza_C2AE_Class_Conditioned_Auto-Encoder_for_Open-Set_Recognition_CVPR_2019_paper.pdf)]
* Lei Shu, Hu Xu and Bing Liu. 2018. **Unseen Class Discovery in Open-world Classification**. arXiv. [[paper](https://arxiv.org/pdf/1801.05609.pdf)] 
* Yang Yu, Wei-Yang Qu, Nan Li, Zimin Guo. 2017.  **Open-Category Classification by Adversarial Sample Generation**. In *Proceedings of IJCAI 2017*. [[paper](https://arxiv.org/abs/1705.08722)] [[code](https://github.com/eyounx/ASG)]
* Manuel Gunther, Steve Cruz, Ethan M. Rudd, Terrance E. Boult. 2017.  **Toward Open-Set Face Recognition**. In *Proceedings of CVPR 2017 Workshop*.[[paper](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w6/papers/Gunther_Toward_Open-Set_Face_CVPR_2017_paper.pdf)] [[code](https://github.com/abhijitbendale/OSDN)]
* Abhijit Bendale and Terrance E. Boult. 2016. **Towards Open Set Deep Networks**. In *Proceedings of CVPR 2016*. [[paper](https://openaccess.thecvf.com/content_cvpr_2016/papers/Bendale_Towards_Open_Set_CVPR_2016_paper.pdf)] [[code](https://github.com/abhijitbendale/OSDN)]
* Abhijit Bendale and Terrance E. Boult . 2015. **Towards Open World Recognition**. In *Proceedings of CVPR 2015*. [[paper](https://openaccess.thecvf.com/content_cvpr_2015/papers/Bendale_Towards_Open_World_2015_CVPR_paper.pdf)] [[code](https://github.com/abhijitbendale/OWR)]
* Walter J. Scheirer, Lalit P. Jain and Terrance E. Boult. 2014. **Probability Models for Open Set Recognition**. *IEEE Transactions on Pattern Analysis and Machine Intelligence*.[[paper](https://ieeexplore.ieee.org/abstract/document/6809169)] 
* Lalit P. Jain, Walter J. Scheirer and Terrance E. Boult. **Multi-class open set recognition using probability of inclusion**. 2014. In *Proceedings of ECCV 2014*. [[paper](https://link.springer.com/chapter/10.1007/978-3-319-10578-9_26)]
* Walter J. Scheirer, Anderson de Rezende, Archana Sapkota and Terrance E. Boult . 2013. **Toward Open Set Recognition**. *IEEE Transactions on Pattern Analysis and Machine Intelligence*. [[paper](https://ieeexplore.ieee.org/abstract/document/6365193)] 

<h3 id="New_Category_Discovery">New Category Discovery</h3> 

* Kai Han∗, Sylvestre-Alvise Rebuffi∗, Sebastien Ehrhardt∗, Andrea Vedaldi and Andrew Zisserman. 2020. **Automatically Discovering and Learning New Visual Categories with Ranking Statistics**. In *Proceedings of ICLR 2020*. [[paper](https://openreview.net/pdf?id=BJl2_nVFPB)] [[code](https://github.com/k-han/AutoNovel)]
* Kai Han, Andrea Vedaldi and Andrew Zisserman. 2019. **Learning to Discover Novel Visual Categories via Deep Transfer Clustering**. In *Proceedings of ICCV 2019*. [[paper](https://www.robots.ox.ac.uk/~vgg/research/DTC/files/iccv2019_DTC.pdf)] [[code](https://github.com/k-han/DTC)]
* Yen-Chang Hsu, Zhaoyang Lv, Joel Schlosser, Phillip Odom and Zsolt Kira. 2019. **Multi-class classification without multi-class labels**. In *Proceedings of ICLR 2019*. [[paper](https://openreview.net/pdf?id=SJzR2iRcK7)] [[code](https://github.com/GT-RIPL/L2C)]
* Yen-Chang Hsu and Zhaoyang Lv and Zsolt Kira. 2018. **Learning to Cluster in Order to Transfer Across Domains and Tasks**.  In *Proceedings of ICLR 2018*. [[paper](https://openreview.net/pdf?id=ByRWCqvT-)] [[code](https://github.com/GT-RIPL/L2C)]
* Yen-Chang Hsu, Zhaoyang Lv and Zsolt Kira. 2016. **Deep Image Category Discovery using a Transferred Similarity Function**. arXiv. [[paper](https://arxiv.org/pdf/1612.01253.pdf)] 


<h2 id="ML">Machine Learning</h2> 

<h3 id="Deep_Clustering">Deep Clustering</h3> 

* Xiaohang Zhan, Jiahao Xie, Ziwei Liu, Yew-Soon Ong and Chen Change Loy. 2020. **Online Deep Clustering for Unsupervised Representation Learning**.  In *Proceedings of CVPR 2020*. [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhan_Online_Deep_Clustering_for_Unsupervised_Representation_Learning_CVPR_2020_paper.pdf)] [[code](https://github.com/open-mmlab/OpenSelfSup)]
* Yuki M. Asano, Christian Rupprecht and Andrea Vedaldi. 2020. **Self-labelling via simultaneous clustering and representation learning**. In *Proceedings of ICLR 2020*. [[paper](https://openreview.net/pdf?id=Hyx-jyBFPr)] [[code](https://github.com/yukimasano/self-label)]
* Tapaswi Makarand, Law Marc T and Fidler Sanja. 2019. **Video Face Clustering with Unknown Number of Clusters**. In *Proceedings of ICCV 2019*. [[paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Tapaswi_Video_Face_Clustering_With_Unknown_Number_of_Clusters_ICCV_2019_paper.pdf)] [[code](https://github.com/makarandtapaswi/BallClustering_ICCV2019)]
* Caron Mathilde, Bojanowski Piotr, Joulin Armand and Douze Matthijs. 2018. **Deep Clustering for Unsupervised Learning of Visual Features**. In *Proceedings of ECCV 2018*. [[paper](https://arxiv.org/pdf/1807.05520.pdf)] [[code](https://github.com/facebookresearch/deepcluster)]
* Chen Shi, Qi Chen, Lei Sha, Sujian Li, Xu Sun, Houfeng Wang and Lintao Zhang. 2018. **Auto-Dialabel: Labeling Dialogue Data with Unsupervised Learning**.  In *Proceedings of EMNLP 2018*. [[paper](https://www.aclweb.org/anthology/D18-1072.pdf)]
* Jianlong Chang, Lingfeng Wang, Gaofeng Meng, Shiming Xiang and Chunhong Pan. 2017. **Deep adaptive image clustering**.  In *Proceedings of ICCV 2017*. [[paper](https://openaccess.thecvf.com/content_ICCV_2017/papers/Chang_Deep_Adaptive_Image_ICCV_2017_paper.pdf)] [[code](https://github.com/vector-1127/DAC)]
* Bo Yang, Xiao Fu, Nicholas D. Sidiropoulos and Mingyi Hong. 2017. **Towards k-means-friendly spaces: Simultaneous deep learning and clustering**.  In *Proceedings of ICML 2017*. [[paper](https://dl.acm.org/doi/10.5555/3305890.3306080)] [[code](https://github.com/boyangumn/DCN)]
* Yen-Chang Hsu and Zsolt Kira. 2016. **Neural network-based clustering using pairwise constraints**. In *Proceedings of ICLR 2016 Workshop*. [[paper](https://arxiv.org/pdf/1511.06321.pdf)] [[code](https://github.com/GT-RIPL/L2C)]
* Jianwei Yang, Devi Parikh and Dhruv Batra. 2016. **Joint Unsupervised Learning of Deep Representations and Image Clusters**. In *Proceedings of CVPR 2016*. [[paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Yang_Joint_Unsupervised_Learning_CVPR_2016_paper.pdf)] [[code](https://github.com/jwyang/JULE.torch)]
* Junyuan Xie, Ross Girshick and Ali Farhadi. 2016. **Unsupervised Deep Embedding for Clustering Analysis**.  In *Proceedings of ICML 2016*. [[paper](https://dl.acm.org/doi/10.5555/3045390.3045442)] [[code](https://github.com/piiswrong/dec)]
* Zhiguo Wang, Haitao Mi and Abraham Ittycheriah. 2016. **Semi-supervised Clustering for Short Text via Deep Representation Learning**.  In *Proceedings of CoNLL 2016*. [[paper](https://www.aclweb.org/anthology/K16-1004.pdf)]
* Jiaming Xu, Peng Wang, Guanhua Tian, Bo Xu, Jun Zhao, Fangyuan Wang and Hongwei Hao. 2015. **Short Text Clustering via Convolutional Neural Networks**.  In *Proceedings of ACL Workshop 2015*. [[paper](https://www.aclweb.org/anthology/W15-1509.pdf)] [[dataset](https://github.com/jacoxu/StackOverflow)]

<!-- <h2 id="Multimodality">Multimodality</h2> 

<h3 id="Dialogue_System">Dialogue System</h3> 

* Amrita Saha, Mitesh M. Khapra and Karthik Sankaranarayanan. 2018. **Towards Building Large Scale Multimodal Domain-Aware Conversation Systems**.  In *Proceedings of AAAI 2018*. [[paper](https://arxiv.org/abs/1704.00200)] [[code](https://github.com/lipiji/dialogue-hred-vhred)] -->
