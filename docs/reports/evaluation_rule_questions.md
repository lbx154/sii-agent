# 评测规则待确认问题

老师好，我有几个评测规则问题想确认，主要是为了避免无意中使用测试集信息或造成不公平比较：

1. 请问 **2Wiki 本身的 train split** 是否允许作为训练数据使用？

2. 反思模块要求智能体在失败后分析失败原因。测试时是否允许直接将模型输出和 **gold answer** 比较来判断失败？如果不允许，测试时应该如何定义“失败”并触发反思？例如是否只能基于无答案、低置信、工具循环、证据不足等模型自身信号？

3. 是否允许在 **测试集问题上做自蒸馏参数微调**？训练数据由模型自己生成，且不使用/不泄露 gold answer 打标签，这种 test-set adaptation 是否允许？

4. 测试时是否允许 **test-time self-evolution / test-time memory update**？比如测试 2Wiki 时，agent 在前面测试样本中产生的反思、经验、memory，是否可以用于后续测试样本？如果允许，是否绝对不能使用答案正确与否的信号？还是说 agent 必须在测试前已经完成进化，测试时只做固定推理？

5. 训练集可以自由选用吗？是否只要不与测试集重合、不包含测试题答案泄露，就可以使用任意公开数据或其他 QA 数据？

6. 推理时间应该如何公平比较？不同推理引擎如 vLLM / SGLang，以及不同并发数、batching、max tokens、thinking 开关、tool parser、GPU 数量等，都会导致 wall time 差异。最终是比较 wall-clock time、平均 token 数、工具调用次数，还是只报告配置即可？

7. LLM-as-judge 是否有官方固定 prompt、模型和解析规则？是否必须逐条用 judge 判，还是 local exact/F1 已正确的样本可以直接计为正确？对于 “unknown / cannot determine / insufficient evidence” 这类回答，官方 judge 是否会严格判错？

8. 对于 BrowseComp/固定语料任务，检索工具是否必须使用官方 BM25 index 和官方 BM25 参数？top-k、snippet 长度、是否允许 get_document、是否允许自建 reranker/embedding rerank，这些是否有限制？

9. 测试时是否允许多次运行同一个样本并做 self-consistency、selector、reranking 或 best-of-N？如果允许，能否用无 gold 的 LLM selector 选择答案？如果不允许，是否每个样本只能一次推理提交？

10. 是否允许先跑一遍测试集，再只对模型答错或低置信的样本重跑？如果“答错”来自 gold/local scorer 显然不允许，那如果只根据模型自身低置信或无答案来选择重跑是否允许？

11. 模型的 thinking/reasoning mode 是否允许自由开启？是否需要统一要求或在结果中明确报告？如果某些模型/引擎的 tool calling 与 thinking 兼容性不同，是否可以做适配解析？

12. 对 LoRA/微调模型，是否只要训练数据合规即可？是否允许在公开训练集或非测试集生成轨迹上训练工具使用、反思策略或检索策略？
