我们的创新性在于利用d3s的优势去设计框架的设计以及选择高校的模块，以保证跟踪效果和时间的平衡。

在这里我们和审稿人也不一样的认知，



We perform experiments on two public evaluation datasets for the analysis not only on all sequences with motion blur, but also the other attributes. The experiments on the public dataset was lacking  in prior work mentioned in the Introduction.



For fair experimetal comparison, 我们在fig 4 和fi5 展示了公共数据集上的效果，并且在table2中单独展现了在模糊序列下的效果，并在exper进行了分析，证明我们的效果并不是平均。

我们也指明了，效果不一样第一是由于lalot具有很长的序列，我们的方法对长时序列有更好的鲁棒性



We have already done the analyses about the phenomenon in paragraph 3.3 .The main reason is that LaSOT has a large number of long-term videos which lead to dramatic changes in target appearance, and our method is more able to reduce the risk of tracking drift on the LaSOT dataset. 


Table 2 shows the results of each attribute including blurred and not just blurred. Our analysis in 3.2.3 and 3.3. our method improves the performance on all cases.



also greatly improves the performance on the
FOC, POC and BC subsets.



 And our analysis of the table 



For fair experimetal comparison, the table2 shows the results of each attributes include of blured cases .And we anlyses in 3.2.3 and 3.3 表明我们的效果不是利用平云。









公共数据集上的分析 缺少在



We 



For experiments, two public evaluation datasets 





Review 0036 我们的实验在两个公共数据集进行了验证，而且同时展示了模糊场景下的性能。而过去的研究仅仅在挑选的个别的模糊序列上进行了测试，并未在公共数据集上进行展示并制作精度图和成功图。



our experimental validation on public subset, at the same time ,showes the performance in the motion blur. The prior method metionded in introduction only evalt on 少数的几个序列上并未showes the precison and success。









我们真诚感谢审稿人的认真阅读、建设性的问题和建议。我们非常希望进一步交流，以改进我们的工作，但以下是我们在目前范围内的最大努力。

We would like to thank all the reviewers for providing valuable feedback. Below are our responses to the comments.

We thank you for your comments. Apparently there is some conceptual confusion hehre: 我们方法的新颖之处在于，我们针对这项任务提出了一个架构，在这个架构里使用最优的方法，满足速度和精度的平衡，并达到较好的效果。



我们的关注点在于利用d3s的优势，制作一个架构，在架构内使用最合适的方法，



而我们并没有对网络进行重新训练，重点并不在网络模型结构的设计上，使用的仍是。。而我们在框架的设计和方法的选择上给出了足够的理由，并给出了清晰的流程以及参数的设计。这与dl的复杂程度无关。

评论者认为我们。。。

而我们并没有对网络进行重新训练，重点并不在网络模型结构的设计上，使用的仍是。。而我们在框架的设计和方法的选择上给出了足够的理由，并在并给出了清晰的流程以及参数的设计



我们在第2 部分给出了框架的设计。并且给出了具体的参数的设计与方法。。我们觉得评论者容易混淆，我们使用的是离线的参数，并没有利用训练提高跟踪器的性能。所以不设计DL

所以，我们主要是为了选择最适合我们框架的算法，有很多iqa的深度学习的算法，虽然精确，但耗时很久我们并未考虑。

deblurgan

The choice of the two de-blurring solutions is not justified. Even if these 2 methods  are new it is important to justify this choice.

我们没有给出理由，我们验证了很多算法，deblurv2由于速度和效率被我们所接受，我们会改进它



所以我们并没有对iqa和deblr模块的结构进行详细的描述。和挑选的必要性。而是直接使用了一种可以确保能不影响速度的方法。

所以我们直接给出了所选择的方法，为了复现论文，我们将需要设置参数的地方设置。

而并没有利用。。来提升效果，所以在deblur和d3s我们使用了离线的参数。



我们并未修改它的结构，它的详细结构可参加参考文献。





我们的目的是提出一个框架用于解决固定场景的问题。所以是好复现的。

#### **1. Importance/Relevance**

|      | **Importance/Relevance** | **Justification of Importance/Relevance Score**              |
| ---- | ------------------------ | ------------------------------------------------------------ |
| R1   | Of broad interest        | None                                                         |
| R2   | Of sufficient interest   | Interesting problem of sufficient interest in various applications. |
| R3   | Of sufficient interest   | None                                                         |

充足有趣



#### **2. Novelty/Originality/Contribution**

|      | **Novelty/Originality/Contribution** | **Justification of Novelty/Originality/Contribution Score**  |
| ---- | ------------------------------------ | ------------------------------------------------------------ |
| R1   | Moderately original                  | Tracking in motion blur case is indeed not well-studied. The authors propose to enhance the D3S model specifically for motion blur tracking. |
| R2   | Minor originality                    | The originality of the authors' contribution does not lie in the new ideas on the algorithmic or conceptual level but essentially in the way they combine existing tools and methods to solve an interesting problem which concerns visual tracking in the context of motion blur. To be honest, I didn't see any significant originality that could make a substantial advancement in the domain. |
| R3   | Moderately original                  | The main novelty of the proposal is the inclusion of an image quality assesment component prior to the deblurring network to judge the level of blur of the image. If the image is not blur, the tracking is performed with a modified version of DS3 network. |



我们是针对目标模糊这项任务，在架构上的创新方式，以确保在时间允许的范围内，使效果达到最好。

适度创新：我觉得我们是结构上的设计创新。使用最好最快的方式





#### **3. Technical Correctness**


|      | **Technical Correctness**     | **Justification of Technical Correctness Score** |
| ---- | ---- | ---- |
| R1   | Probably correct | None |
| R2   | Probably correct | It is uneasy to check wether the results are correct or not in such complex architecture based on DL. This is the case of so many similar methods based on DL approach.<br /> Furthermore in the absence of the implementation details it is impossible to check the technical correctness of the method. |
| R3   | Definitely correct | None |

完全正确

由于我们是结构上的设计，  阈值我们已经写的很清楚了。软件流程我们在方法上已经给出






#### **4.Experimental Validation**


|      | **Experimental Validation**     | **Justification of Technical Correctness Score** |
| ---- | ---- | ---- |
| R1   | Limited but convincing | The experimental validation seems satisfactory for the proposed approach. I understand that there is a limited number of existing studies that focus on tracking in motion blur, but I wonder how the proposed methodology perform compared to these methods (mentioned in the Introduction)? |
| R2   | Limited but convincing | Using BRISQUE as an index for detecting and measuring the level of blur is not really the best solution. It would be better to use some known dedicated blur detection and evaluation metrics. It would be useful for reader to provide more details on the de-blurring process for the sake of completeness instead of just citing the reference. Because this module is critical. Furthermore, the provided implementations details are insufficient to allow reader to reproduce the results.  <br />The choice of the two de-blurring solutions is not justified. Even if these 2 methods  are new it is important to justify this choice. |
| R3   | Sufficient validation/theoretical paper | None |

解释清楚

R1：其他算法性能都太差了，我们用了

R2: 我们主要考虑速度方面的问题，很多去模糊算法都及其耗时，




#### **5.Clarity of Presentation**

|      |  **Clarity of Presentation**    | **Justification of Clarity of Presentation Score** |
| ---- | ---- | ---- |
| R1   | Very clear | None |
| R2   | Difficult to read | Figures 4 and 5 are completely unreadable. should be redone (zoomed).  <br />Some sentences need to be rephrased. IQA is misspelled in page 2. <br />Fig.3 axis labels are missing. <br />Ablation part needs to be rewritten (style and some grammar errors); the same remark applies for some paragraphs. The authors are urged to improve the scientific writting. It is acceptable but it would be better to improve some parts. |
| R3   | Clear enough | There are some typos in the paper. <br /><br /> Section 2. "...processing time. The context region two point five times the target size". -> Which is the meaning of this sentence?. where is the verb? <br />Section 2.2 "... by the presence of distortion, we count BRISQUE..." -> Should be "... by the presence of distortion. We count BRISQUE..."  <br /><br />Figures 4 and 5 are too small, they are unreadable in printed version.  <br /><br />Section 3.1. "Our tracker reaches a speed of 10fps" -> at which resolution?, please specify. |

比较清楚  更正就行



#### **6.Reference to Prior Work**


|      | **Reference to Prior Work**     | **Justification of Reference to Prior Work** |
| ---- | ---- | ---- |
| R1   | References adequate | None |
| R2   | References adequate | None |
| R3   | References adequate | None |


#### **7.Additional comments to author(s)**


|      | **Additional comments to author(s)**                         |
| ---- | ------------------------------------------------------------ |
| R1   | Considering overall performance evaluation, it seems that the proposed approach with the enhanced D3S outperforms other approaches in Fig.4 and produces slightly better results in Fig.5. How could you explain that? Does the performance improvement come from the robustness to blurred cases or does your approach produce higher performances on average (not just for motion blur)? |
| R2   | None                                                         |
| R3   | None                                                         |

R1:我们已经解释清楚。