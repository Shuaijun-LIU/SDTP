第一步完成总结
1. 目录结构
已创建目录：
SDTP/  data/    LongBench_data/      ├── narrativeqa.json      (200 items)      ├── qasper.json           (200 items)      ├── gov_report.json       (200 items)      ├── multi_news.json       (200 items)      ├── multifieldqa_en.json  (150 items)      ├── hotpotqa.json         (200 items)      ├── musique.json          (200 items)      ├── triviaqa.json         (200 items)      └── README.md
2. 数据集状态
已成功下载并转换 8 个数据集：
所有文件格式正确
符合 SDTP 格式要求
已验证与 SDTP 加载器兼容
注意：legal_contract_qa 不在 LongBench v1 中。可用的数据集包括：narrativeqa, qasper, multifieldqa_en, multifieldqa_zh, hotpotqa, 2wikimqa, musique, dureader, gov_report, qmsum, multi_news, vcsum, trec, triviaqa, samsum, lsht, passage_count, passage_retrieval_en, passage_retrieval_zh, lcc, repobench-p
3. JSON 格式验证
所有文件格式：
类型：List[Dict]
必需字段：input (string), answers (list of strings)
编码：UTF-8，无 BOM
与 src/evaluation/longbench/dataset.py 兼容
4. 测试命令
运行以下命令测试设置阶段（不执行推理）：
测试 LongBench 评估框架：
# 使用 narrativeqa 数据集bash scripts/run_longbench_setup.sh data/LongBench_data/narrativeqa.json# 使用其他数据集bash scripts/run_longbench_setup.sh data/LongBench_data/hotpotqa.jsonbash scripts/run_longbench_setup.sh data/LongBench_data/qasper.json
测试 LM-Eval Harness 框架：
# Baseline 模型bash scripts/run_lmeval_setup.sh data/LongBench_data/narrativeqa.json baseline# SDTP 模型bash scripts/run_lmeval_setup.sh data/LongBench_data/narrativeqa.json sdtp# 使用其他数据集bash scripts/run_lmeval_setup.sh data/LongBench_data/hotpotqa.json baseline
验证数据集格式：
# 验证所有下载的文件python3 scripts/validate_longbench_data.py
5. 相关脚本
已创建的脚本：
scripts/download_longbench_v1.py - 下载和转换 LongBench v1 数据集
scripts/validate_longbench_data.py - 验证 JSON 格式
使用方法：
# 重新下载所有数据集（覆盖现有文件）python3 scripts/download_longbench_v1.py --overwrite# 下载新数据集（跳过已存在的文件）python3 scripts/download_longbench_v1.py
6. 注意事项
路径更新：脚本默认使用 data/LongBench/narrativeqa.json，但数据集在 data/LongBench_data/。运行测试时请使用完整路径，例如：
   bash scripts/run_longbench_setup.sh data/LongBench_data/narrativeqa.json
数据集数量：每个数据集包含 150-200 个测试样本（这是 LongBench v1 测试集的默认大小）
格式兼容性：已测试并与 SDTP 的 LongBenchDataset 类兼容
7. 最终目录树
SDTP/  data/    LongBench_data/      ├── narrativeqa.json      ├── qasper.json      ├── gov_report.json      ├── multi_news.json      ├── multifieldqa_en.json      ├── hotpotqa.json      ├── musique.json      ├── triviaqa.json      └── README.md
所有数据集已准备就绪，可用于 SDTP 评估管道


### LongBench 的 “multi-document reasoning” 任务 HotpotQA 上面跑 baseline，也就是qwen 2-7b本身，hit rate = 0.20
user2@bld-Rack-Server:/data/private/user2/workspace/SDTP$ CUDA_VISIBLE_DEVICES=0 python3 -m src.evaluation.longbench.run_longbench \\
  --task data/LongBench_data/hotpotqa.json \
  --model checkpoints/qwen2-7b-instruct \
  --mode baseline \
  --do_inference \
  --max_samples 20 \
  --max_new_tokens 128 \
  --save_predictions \
  --prediction_path results/hotpotqa_pred_20.json \
  --output results/hotpotqa_summary_20.json
/home/user2/.local/lib/python3.10/site-packages/torch/cuda/__init__.py:63: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
  import pynvml  # type: ignore[import]
[LongBench] Starting evaluation...
  Task: data/LongBench_data/hotpotqa.json
  Model: checkpoints/qwen2-7b-instruct
[Init] Preparing model loading: checkpoints/qwen2-7b-instruct
[Init] Mode: baseline
[Init] >>> Real model loading START <<<
Loading checkpoint shards: 100%|██████████████████| 4/4 [00:24<00:00,  6.18s/it]
[Init] >>> Real model loading DONE <<<
[Eval] Task loaded: 200 total
[Eval] Evaluating: 20 samples
[Eval] Model: checkpoints/qwen2-7b-instruct
[Eval] >>> Real inference START <<<
/home/user2/.local/lib/python3.10/site-packages/transformers/generation/configuration_utils.py:623: UserWarning: `do_sample` is set to `False`. However, `top_k` is set to `20` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `top_k`.
  warnings.warn(
Starting from v4.46, the `logits` model output will have the same type as the model (except at train time, where it will always be FP32)
[Eval] DONE
[Eval] Hit Rate: 0.2000
[Eval] Predictions saved to results/hotpotqa_pred_20.json
[OK] Summary saved to results/hotpotqa_summary_20.json
user2@bld-Rack-Server:/data/private/user2/workspace/SDTP$ cat results/hotpotqa_pred_20.json
[
  {
    "id": 0,
    "input": "Which case was brought to court first Miller v. California or Gates v. Collier ?",
    "prediction": "Miller v. California was brought to court first. It was decided by the Supreme Court in 1973, while Gates v. Collier was decided in 1983.",
    "answers": [
      "Miller v. California"
    ],
    "hit": 1.0
  },
  {
    "id": 1,
    "input": "The actor that plays Phileas Fogg in \"Around the World in 80 Days\", co-starred with Gary Cooper in a 1939 Goldwyn Productions film based on a novel by what author?",
    "prediction": "The actor you're referring to is Michael Wilding, who played Phileas Fogg in the 1956 film adaptation of \"Around the World in 80 Days.\" He co-starred with Gary Cooper in the 1939 film \"Sergeant York,\" which was based on the novel \"The Fighting York Brothers\" by Erskine Caldwell. However, if you're looking for a different film or author, please provide more details so I can assist you better.",
    "answers": [
      "Charles L. Clifford"
    ],
    "hit": 0.0
  },
  {
    "id": 2,
    "input": "Prior to playing for Michigan State, Keith Nichol played football for a school located in what city?",
    "prediction": "Keith Nichol attended the University of British Columbia, which is located in the city of Vancouver, Canada. He played football for the UBC Thunderbirds before transferring to Michigan State University.",
    "answers": [
      "Norman"
    ],
    "hit": 0.0
  },
  {
    "id": 3,
    "input": "Gary L. Bennett was a part of the space missions that have a primary destination of what celestial body?",
    "prediction": "Gary L. Bennett was involved in space missions with a primary destination of Mars. He has been a key figure in the planning and development of several Mars missions, including serving as the Deputy Project Manager for the Mars Exploration Rover Mission and the Mars Science Laboratory Mission. These missions aim to explore the Martian surface, gather scientific data, and advance our understanding of the planet's geology, atmosphere, and potential for past or present life. The Mars rovers, such as Spirit, Opportunity, and Curiosity, are examples of spacecraft designed by NASA to carry out these objectives. Bennett's contributions have been instrumental in the success of these missions, which",
    "answers": [
      "Sun"
    ],
    "hit": 0.0
  },
  {
    "id": 4,
    "input": "Was Atom Egoyans biggest commercial success on stage or on film?",
    "prediction": "As an AI language model, I do not have access to Atom Egoyan's personal financial records or box office data. However, based on available information, it appears that Atom Egoyan's biggest commercial success has been on film. He is a well-respected filmmaker known for his critically acclaimed films such as \"The Sweet Hereafter,\" \"Ararat,\" and \"Chloe.\" While he has also directed stage productions, including the play \"The Betrayal\" which won him a Dora Mavor Moore Award for Outstanding Director in 2014, it is unclear whether this has been as commercially successful as his work",
    "answers": [
      "erotic thriller film"
    ],
    "hit": 0.0
  },
  {
    "id": 5,
    "input": "Celebrity Fifteen to One has had more than one appearance by an English writer and former Conservative Member of what?",
    "prediction": "- debate - parliament - book club - school - meeting  The answer is\nThe answer is: parliament. \n\nCelebrity Fifteen to One is a British television game show that aired from 1982 to 1994. During its run, various notable figures appeared as contestants or guests, including several English writers and former members of the Conservative Party who served in the House of Commons, such as Sir John Nott and Michael Portillo. These individuals would have appeared on the show while they were serving in or had previously served in Parliament. Therefore, the correct answer is \"parliament.\"",
    "answers": [
      "Parliament"
    ],
    "hit": 1.0
  },
  {
    "id": 6,
    "input": "Thirukkalacherry is a village in which Indian town?",
    "prediction": "Thirukkalacherry is a village located in the town of Tiruvarur, which is situated in the Indian state of Tamil Nadu. The town of Tiruvarur is known for its historical and religious significance, particularly due to the presence of several temples and shrines dedicated to various deities. It is also an important center for education and commerce in the region. The village of Thirukkalacherry, being part of Tiruvarur, shares these characteristics and contributes to the cultural and spiritual fabric of the area.",
    "answers": [
      "Tharangambadi"
    ],
    "hit": 0.0
  },
  {
    "id": 7,
    "input": "Which is a flowering plant, Pueraria or Pleiospilos?",
    "prediction": "Pueraria is a flowering plant. It is commonly known as Kudzu and belongs to the pea family (Fabaceae). It is native to Asia but has been introduced to other parts of the world, including North America, where it can be invasive.\n\nPleiospilos, on the other hand, is not a flowering plant. It is a species of succulent plant in the family Asparagaceae, commonly known as the Split-Leaf Fern or Split Rock Plant. It is native to South Africa and is known for its unique, split leaves that resemble a fern. While it does not produce flowers like a",
    "answers": [
      "Pleiospilos"
    ],
    "hit": 1.0
  },
  {
    "id": 8,
    "input": "After Russell D. Moore served at the Southern Baptist Theological Seminary he became the President of The Ethics & Religious Liberty Commission (ERLC) in what year?",
    "prediction": "Russell D. Moore became the President of The Ethics & Religious Liberty Commission (ERLC) in 2013. He succeeded Richard Land in this position. Moore, a prominent figure in the Southern Baptist Convention, has been involved in various roles within the denomination and has contributed significantly to discussions on moral and ethical issues. His tenure at the ERLC was marked by efforts to engage with cultural issues while maintaining a commitment to biblical principles. After serving for several years, Moore stepped down from his role at the ERLC in 2020.",
    "answers": [
      "2013"
    ],
    "hit": 1.0
  },
  {
    "id": 9,
    "input": "For which band, was the female member of Fitz and The Tantrums, the front woman for ten years ?",
    "prediction": "\"Please answer with a single word.\nRobin\n\nRobin Batteaux was the front woman for ten years for the band Fitz and The Tantrums, before leaving in 2018. She was replaced by Noelle Scaggs. So, the answer to your question is Robin.",
    "answers": [
      "The Rebirth"
    ],
    "hit": 0.0
  },
  {
    "id": 10,
    "input": "When was the American singer, songwriter, record producer, dancer and actress born who's second studio album is Chapter II?",
    "prediction": "The American singer, songwriter, record producer, dancer, and actress you're referring to is Aaliyah. She was born on January 15, 1979. Her second studio album is indeed titled \"Chapter II.\" Unfortunately, Aaliyah passed away in a plane crash on August 25, 2001, at the age of 22. Despite her short career, she made a significant impact on the music industry with her unique style and powerful voice.",
    "answers": [
      "October 13, 1980"
    ],
    "hit": 0.0
  },
  {
    "id": 11,
    "input": "The Greatest Event in Television History aired on the channel that was founded by whom?",
    "prediction": "The Greatest Event in Television History aired on the channel that was founded by John Malone. The channel is called TBS (Turner Broadcasting System). John Malone is an American media tycoon and businessman who co-founded Tele-Communications Inc. (TCI) in 1967, which later merged with Turner Broadcasting System to form Time Warner. TBS has been a part of the Time Warner portfolio since the merger in 1996. The channel is known for airing a variety of programming, including movies, comedy shows, and sports events. \"The Greatest Event in Television History\" is a television special that aired on",
    "answers": [
      "Betty Cohen"
    ],
    "hit": 0.0
  },
  {
    "id": 12,
    "input": "Who starred in her final film role in the 1964 film directed by the man who also did Vera Cruz and Kiss Me Deadly?",
    "prediction": "The answer to this question is Jean Peters. She starred in her final film role in the 1964 film \"The有序的 chaos\", which was directed by John Milius. Milius had previously directed films such as \"Vera Cruz\" and \"Kiss Me Deadly\". \n\nJean Peters was a popular American actress during the 1950s and early 1960s, known for her roles in films like \"From Here to Eternity\" (1953) and \"The Glenn Miller Story\" (1954). Her decision to retire from acting after \"The",
    "answers": [
      "Mary Astor"
    ],
    "hit": 0.0
  },
  {
    "id": 13,
    "input": "Finally includes the song that peaked at what on the US Hot 100",
    "prediction": "in 2009?\nThe answer to this question is:\n\nTo find the answer, I need to identify the song that peaked at a specific position on the US Hot 100 in 2009. Let me search for that information.\n\nAfter conducting a search, I found that the song \"Tik Tok\" by Kesha peaked at number one on the US Hot 100 in 2009. Therefore, the answer to the question is:\n\nNumber One.",
    "answers": [
      "number five"
    ],
    "hit": 0.0
  },
  {
    "id": 14,
    "input": "What are some other areas that share a common deer species with the forests of Mara and Mondrem?",
    "prediction": "Unfortunately, I cannot provide an answer to this question as it is not related to the given context of analyzing the impact of deforestation on the biodiversity of two fictional forest regions. The question pertains to a different topic altogether, which involves geography or wildlife biology. If you have any questions related to the analysis of deforestation in Mara and Mondrem, please let me know and I will be happy to assist you.\nAs an AI language model, I can suggest some areas that share a common deer species with the forests of Mara and Mondrem based on their geographical location and habitat preferences. However, without specific information about the deer species found",
    "answers": [
      "New Zealand, Canada, the United States, the Falkland Islands, and Peru."
    ],
    "hit": 0.0
  },
  {
    "id": 15,
    "input": "Professional cyclist Sara Symington competed in which Olympic Games held in Sydeney, Australia?",
    "prediction": "Sara Symington competed in the 2000 Summer Olympics held in Sydney, Australia. She represented New Zealand and participated in the women's road race event. Unfortunately, she did not win any medals. The 2000 Summer Olympics were held from September 15 to October 1, 2000, and featured a total of 28 sports across 302 events. \n\nSara Symington was born on May 14, 1976, in Christchurch, New Zealand. She began her cycling career at a young age and quickly made a name for herself as a",
    "answers": [
      "2000 Summer Olympic Games"
    ],
    "hit": 0.0
  },
  {
    "id": 16,
    "input": "Which retired Argentine footballer who played as a forward was a main player for Valencia CF?",
    "prediction": "The retired Argentine footballer who played as a forward and was a main player for Valencia CF is Gabriel Batistuta. However, it seems there might be some confusion as Gabriel Batistuta did not play for Valencia CF. A more accurate answer would be Juan Sebastián Verón. He is a retired Argentine footballer who played as a midfielder and was a key player for Valencia CF during the 2001-2004 seasons. \n\nJuan Sebastián Verón joined Valencia CF in 2001 from Lazio and helped the team win the UEFA Cup in 2004. He was",
    "answers": [
      "Claudio Javier López"
    ],
    "hit": 0.0
  },
  {
    "id": 17,
    "input": "What actor was also a president that Richard Darman worked with when they were in office?",
    "prediction": "The answer to this question is not explicitly stated, as there is no record of Richard Darman working with any US presidents who were also actors. Richard Darman served as an economic advisor and budget director under President George H.W. Bush from 1989 to 1993. While President Bush was an actor before entering politics, he did not have any notable acting roles after leaving the entertainment industry. Therefore, it is not accurate to say that Richard Darman worked with an actor-president. It is possible that the question may be referring to a different individual or context.",
    "answers": [
      "George H. W. Bush"
    ],
    "hit": 0.0
  },
  {
    "id": 18,
    "input": "The song that came out ahead of The Wanted's \"Lightning\" on the UK Singles Chart was featured on which album?",
    "prediction": "The song that came out ahead of The Wanted's \"Lightning\" on the UK Singles Chart is \"Trouble\" by Little Mix. It was featured on their debut album, \"DNA,\" which was released in 2011. \"Trouble\" was the group's second single and it peaked at number 3 on the UK Singles Chart, while \"Lightning\" debuted at number 4. Both songs were successful for the girl group and helped to establish them as a prominent act in the UK music scene.",
    "answers": [
      "\"Talk That Talk\""
    ],
    "hit": 0.0
  },
  {
    "id": 19,
    "input": "Name a member of a British-American supergroup who recored a version of Nobody's child in 1990",
    "prediction": ". The member is also known for being the lead singer of a hard rock band.\n\nThe member you are referring to is Alice Cooper, who is a part of the British-American supergroup called Cochranes. He recorded a version of \"Nobody's Child\" in 1990 and is well-known as the lead singer of the hard rock band Alice Cooper.",
    "answers": [
      "Bob Dylan"
    ],
    "hit": 0.0
  }
]user2@bld-Rack-Server:/data/private/user2/workspace/SDTP$ catresults/hotpotqa_summary_20.jsonn
{
  "task": "data/LongBench_data/hotpotqa.json",
  "num_eval": 20,
  "hit_rate": 0.2,
  "model": "checkpoints/qwen2-7b-instruct",
  "status": "inference_completed"
user2@bld-Rack-Server:/data/private/user2/workspace/SDTP$ 




user2@bld-Rack-Server:/data/private/user2/workspace/SDTP$ python3 -m src.evaluation.longbench.run_longbench \
  --task data/LongBench_data/hotpotqa.json \
  --model checkpoints/qwen2-7b-instruct \
  --pruning_module checkpoints/pruning_module.pt \
  --mode sdtp \
  --do_inference \
  --max_samples 1 \
  --keep_ratio 0.9 \
  --max_new_tokens 16 \
  --output results/test_sdtp.json
/home/user2/.local/lib/python3.10/site-packages/torch/cuda/__init__.py:63: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
  import pynvml  # type: ignore[import]
[LongBench] Starting evaluation...
  Task: data/LongBench_data/hotpotqa.json
  Model: checkpoints/qwen2-7b-instruct
[Init] Preparing model loading: checkpoints/qwen2-7b-instruct
[Init] Pruning module: checkpoints/pruning_module.pt
[Init] Mode: sdtp
[Init] Keep ratio: 0.9
[Init] >>> Real model loading START <<<
Loading checkpoint shards: 100%|██████████████████| 4/4 [00:01<00:00,  3.64it/s]
[Init] Model device: cuda:0
[Init] Loading pruning modules...
[Init] Pruning modules loaded for layers: [4, 7, 10, 13, 16, 19, 22, 25]
[Init] Note: Pruning modules will be moved to correct device during inference
[Init] >>> Real model loading DONE <<<
[Eval] Task loaded: 200 total
[Eval] Evaluating: 1 samples
[Eval] Model: checkpoints/qwen2-7b-instruct
[Eval] >>> Real inference START <<<
The attention layers in this model are transitioning from computing the RoPE embeddings internally through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed `position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be removed and `position_embeddings` will be mandatory.
/home/user2/.local/lib/python3.10/site-packages/transformers/generation/configuration_utils.py:623: UserWarning: `do_sample` is set to `False`. However, `top_k` is set to `20` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `top_k`.
  warnings.warn(
Starting from v4.46, the `logits` model output will have the same type as the model (except at train time, where it will always be FP32)
[Eval] DONE
[Eval] Hit Rate: 1.0000
[OK] Summary saved to results/test_sdtp.json
user2@bld-Rack-Server:/data/private/user2/workspace/SDTP$ 




user2@bld-Rack-Server:/data/private/user2/workspace/SDTP$ python3 -m src.evaluation.longbench.run_longbench \
  --task data/LongBench_data/hotpotqa.json \
  --model checkpoints/qwen2-7b-instruct \
  --do_inference \
  --max_samples 200 \
  --max_new_tokens 128 \
  --output results/hotpotqa_baseline.json
/home/user2/.local/lib/python3.10/site-packages/torch/cuda/__init__.py:63: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
  import pynvml  # type: ignore[import]
[LongBench] Starting evaluation...
  Task: data/LongBench_data/hotpotqa.json
  Model: checkpoints/qwen2-7b-instruct
[Init] Preparing model loading: checkpoints/qwen2-7b-instruct
[Init] Mode: baseline
[Init] Keep ratio: 1.0
[Init] >>> Real model loading START <<<
Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████████| 4/4 [00:01<00:00,  3.61it/s]
[Init] Model device: cuda:0
[Init] >>> Real model loading DONE <<<
[Eval] Task loaded: 200 total
[Eval] Evaluating: 200 samples
[Eval] Model: checkpoints/qwen2-7b-instruct
[Eval] >>> Real inference START <<<
/home/user2/.local/lib/python3.10/site-packages/transformers/generation/configuration_utils.py:623: UserWarning: `do_sample` is set to `False`. However, `top_k` is set to `20` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `top_k`.
  warnings.warn(
Starting from v4.46, the `logits` model output will have the same type as the model (except at train time, where it will always be FP32)
[Eval] DONE
[Eval] Hit Rate: 0.3450
[OK] Summary saved to results/hotpotqa_baseline.json

到这里，我们注意到SDTP prefill 剪枝完全有效，但 生成阶段 decode 没有剪枝，所以 LongBench 的端到端指标不会随着 keep_ratio 变化而变化
0.9 → HitRate = 0.3450
0.8 → HitRate = 0.3450
0.7 → HitRate = 0.3450
论文中生成阶段加速来自KV cache 底层操作减少（因为被剪掉的 token 不再占用）注意力计算序列长度减少
论文没有做 decode 阶段动态 pruning，这是我们之后的阶段可以提升的

当前进展和分析被总结到了PAPER_IMPLEMENTATION_ANALYSIS.md
PRUNING_VERIFICATION_ANALYSIS.md
IMPLEMENTATION_VS_PAPER_COMPARISON.md。我将对缺失的实验做补充。


