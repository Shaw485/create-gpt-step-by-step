# M021 SFT v7.1 Canary 生成评估

模式：`canary`；checkpoint step：`350`；质量门：**FAIL**

| 指标 | Train (64) | 未见问法 Dev/Selection (16) |
|---|---:|---:|
| 严格答案匹配（门控） | 54 (84.4%) | 10 (62.5%) |
| 关键点通过 | 57 (89.1%) | 11 (68.8%) |
| EOS 停止 | 64 (100.0%) | 16 (100.0%) |
| 严重重复 | 0 | 0 |
| 关系方向错误 | 2 | 1 |

## 质量门

| Gate | 结果 |
|---|---|
| `train_exact_answer_rate` | FAIL |
| `holdout_exact_answer_rate` | FAIL |
| `holdout_eos_count` | PASS |
| `per_fact_train_exact_min` | FAIL |
| `per_fact_holdout_exact_min` | FAIL |
| `severe_repetition_count` | PASS |
| `self_relation_error_count` | FAIL |

## 每个 catalog fact_id

| fact_id | Train exact | Holdout exact | Train required terms（诊断） | Holdout required terms（诊断） |
|---|---:|---:|---:|---:|
| `fanjue_identity` | 7/8 | 1/2 | 7/8 | 1/2 |
| `xiaoyan_identity` | 4/8 | 0/2 | 7/8 | 1/2 |
| `xiaozhan_identity` | 7/8 | 2/2 | 7/8 | 2/2 |
| `yaochen_identity` | 7/8 | 1/2 | 7/8 | 1/2 |
| `yaolao_teacher` | 8/8 | 1/2 | 8/8 | 1/2 |
| `yaolao_yaochen_alias` | 5/8 | 1/2 | 5/8 | 1/2 |
| `yihuo_role` | 8/8 | 2/2 | 8/8 | 2/2 |
| `yunlanzong_identity` | 8/8 | 2/2 | 8/8 | 2/2 |

## 全部样本

| Split | fact_id | 问题 | 参考答案 | 生成答案 | Exact | Keypoint | EOS | 重复 | 关系错误 |
|---|---|---|---|---|---|---|---|---|---|
| train | `xiaoyan_identity` | 萧炎是谁？ | 萧炎是萧战的儿子。 | 萧炎是萧战的儿子。 | PASS | PASS | 是 | 否 | 否 |
| train | `xiaoyan_identity` | 请简单介绍一下萧炎。 | 萧炎是萧战的儿子。 | 萧炎是萧战的儿子。 | PASS | PASS | 是 | 否 | 否 |
| train | `xiaoyan_identity` | 萧炎与萧战是什么关系？ | 萧炎是萧战的儿子。 | 萧战是萧家族长，也是萧炎的父亲。 | FAIL | PASS | 是 | 否 | 否 |
| train | `xiaoyan_identity` | 用一句话说明萧炎的家庭身份。 | 萧炎是萧战的儿子。 | 萧炎是萧战的儿子。 | PASS | PASS | 是 | 否 | 否 |
| train | `xiaoyan_identity` | 萧炎是萧战的父亲吗？请纠正。 | 萧炎是萧战的儿子。 | 萧战是萧战的儿子。 | FAIL | FAIL | 是 | 否 | 是 |
| train | `xiaoyan_identity` | 萧炎是谁的儿子？ | 萧炎是萧战的儿子。 | 萧炎是萧战的儿子。 | PASS | PASS | 是 | 否 | 否 |
| train | `xiaoyan_identity` | 萧战的儿子是谁？ | 萧炎是萧战的儿子。 | 萧战是萧家族长，也是萧炎的父亲。 | FAIL | PASS | 是 | 否 | 否 |
| train | `xiaoyan_identity` | 判断正误并说明：萧炎和萧战没有亲属关系。 | 萧炎是萧战的儿子。 | 萧战是萧家族长，也是萧炎的父亲。 | FAIL | PASS | 是 | 否 | 否 |
| train | `xiaozhan_identity` | 萧战是谁？ | 萧战是萧家族长，也是萧炎的父亲。 | 萧战是萧家族长，也是萧炎的父亲。 | PASS | PASS | 是 | 否 | 否 |
| train | `xiaozhan_identity` | 请简单介绍萧战的身份。 | 萧战是萧家族长，也是萧炎的父亲。 | 萧战是萧家族长，也是萧炎的父亲。 | PASS | PASS | 是 | 否 | 否 |
| train | `xiaozhan_identity` | 萧战在萧家是什么身份？ | 萧战是萧家族长，也是萧炎的父亲。 | 萧战是萧家族长，也是萧炎的父亲。 | PASS | PASS | 是 | 否 | 否 |
| train | `xiaozhan_identity` | 萧炎的父亲是谁？ | 萧战是萧家族长，也是萧炎的父亲。 | 萧战是萧家族长，也是萧炎的父亲。 | PASS | PASS | 是 | 否 | 否 |
| train | `xiaozhan_identity` | 萧战是萧炎的儿子吗？请纠正。 | 萧战是萧家族长，也是萧炎的父亲。 | 萧战是萧战的儿子。 | FAIL | FAIL | 是 | 否 | 是 |
| train | `xiaozhan_identity` | 萧战与萧炎是什么关系？ | 萧战是萧家族长，也是萧炎的父亲。 | 萧战是萧家族长，也是萧炎的父亲。 | PASS | PASS | 是 | 否 | 否 |
| train | `xiaozhan_identity` | 从萧炎的角度看，萧战是什么亲属？ | 萧战是萧家族长，也是萧炎的父亲。 | 萧战是萧家族长，也是萧炎的父亲。 | PASS | PASS | 是 | 否 | 否 |
| train | `xiaozhan_identity` | 判断正误并说明：萧战与萧炎没有父子关系。 | 萧战是萧家族长，也是萧炎的父亲。 | 萧战是萧家族长，也是萧炎的父亲。 | PASS | PASS | 是 | 否 | 否 |
| train | `yaochen_identity` | 药尘是谁？ | 药尘就是药老，也是萧炎的老师。 | 药尘就是药老，也是萧炎的老师。 | PASS | PASS | 是 | 否 | 否 |
| train | `yaochen_identity` | 请简要介绍药尘。 | 药尘就是药老，也是萧炎的老师。 | 药尘就是药老，也是萧炎的老师。 | PASS | PASS | 是 | 否 | 否 |
| train | `yaochen_identity` | 药尘还有什么常用称呼？ | 药尘就是药老，也是萧炎的老师。 | 药尘就是药老，也是萧炎的老师。 | PASS | PASS | 是 | 否 | 否 |
| train | `yaochen_identity` | 药尘与萧炎是什么关系？ | 药尘就是药老，也是萧炎的老师。 | 药尘就是药老，也是萧炎的老师。 | PASS | PASS | 是 | 否 | 否 |
| train | `yaochen_identity` | 药尘与药老毫无关系吗？请纠正。 | 药尘就是药老，也是萧炎的老师。 | 药尘就是药老，也是萧炎的老师。 | PASS | PASS | 是 | 否 | 否 |
| train | `yaochen_identity` | 药尘是谁的老师？ | 药尘就是药老，也是萧炎的老师。 | 药尘就是药老，也是萧炎的老师。 | PASS | PASS | 是 | 否 | 否 |
| train | `yaochen_identity` | 萧炎的老师药老原名是什么？ | 药尘就是药老，也是萧炎的老师。 | 药老是萧炎的老师。 | FAIL | FAIL | 是 | 否 | 否 |
| train | `yaochen_identity` | 判断正误并说明：药尘和药老是两个人。 | 药尘就是药老，也是萧炎的老师。 | 药尘就是药老，也是萧炎的老师。 | PASS | PASS | 是 | 否 | 否 |
| train | `yaolao_yaochen_alias` | 药老和药尘是什么关系？ | 药尘是药老曾经的名字，二者是同一人物。 | 药尘就是药老，也是萧炎的老师。 | FAIL | FAIL | 是 | 否 | 否 |
| train | `yaolao_yaochen_alias` | 药尘和药老是否指同一个人？ | 药尘是药老曾经的名字，二者是同一人物。 | 药尘就是药老，也是萧炎的老师。 | FAIL | FAIL | 是 | 否 | 否 |
| train | `yaolao_yaochen_alias` | 药老曾经叫什么名字？ | 药尘是药老曾经的名字，二者是同一人物。 | 药老是萧炎的老师。 | FAIL | FAIL | 是 | 否 | 否 |
| train | `yaolao_yaochen_alias` | 说明药老与药尘这两个称呼的对应关系。 | 药尘是药老曾经的名字，二者是同一人物。 | 药尘是药老曾经的名字，二者是同一人物。 | PASS | PASS | 是 | 否 | 否 |
| train | `yaolao_yaochen_alias` | 有人说药老和药尘是两个人，请纠正。 | 药尘是药老曾经的名字，二者是同一人物。 | 药尘是药老曾经的名字，二者是同一人物。 | PASS | PASS | 是 | 否 | 否 |
| train | `yaolao_yaochen_alias` | 从药尘这个名字看，他后来通常被称为什么？ | 药尘是药老曾经的名字，二者是同一人物。 | 药尘是药老曾经的名字，二者是同一人物。 | PASS | PASS | 是 | 否 | 否 |
| train | `yaolao_yaochen_alias` | 从药老这个称呼反推，他曾经叫什么？ | 药尘是药老曾经的名字，二者是同一人物。 | 药尘是药老曾经的名字，二者是同一人物。 | PASS | PASS | 是 | 否 | 否 |
| train | `yaolao_yaochen_alias` | 判断正误并说明：药老只是药尘的朋友。 | 药尘是药老曾经的名字，二者是同一人物。 | 药尘是药老曾经的名字，二者是同一人物。 | PASS | PASS | 是 | 否 | 否 |
| train | `yaolao_teacher` | 药老和萧炎是什么关系？ | 药老是萧炎的老师。 | 药老是萧炎的老师。 | PASS | PASS | 是 | 否 | 否 |
| train | `yaolao_teacher` | 药老是萧炎的什么人？ | 药老是萧炎的老师。 | 药老是萧炎的老师。 | PASS | PASS | 是 | 否 | 否 |
| train | `yaolao_teacher` | 萧炎与药老是什么师承关系？ | 药老是萧炎的老师。 | 药老是萧炎的老师。 | PASS | PASS | 是 | 否 | 否 |
| train | `yaolao_teacher` | 萧炎的老师是谁？ | 药老是萧炎的老师。 | 药老是萧炎的老师。 | PASS | PASS | 是 | 否 | 否 |
| train | `yaolao_teacher` | 萧炎是药老的老师吗？请纠正。 | 药老是萧炎的老师。 | 药老是萧炎的老师。 | PASS | PASS | 是 | 否 | 否 |
| train | `yaolao_teacher` | 药老教导的弟子是谁？ | 药老是萧炎的老师。 | 药老是萧炎的老师。 | PASS | PASS | 是 | 否 | 否 |
| train | `yaolao_teacher` | 萧炎师从哪位人物？ | 药老是萧炎的老师。 | 药老是萧炎的老师。 | PASS | PASS | 是 | 否 | 否 |
| train | `yaolao_teacher` | 判断正误并说明：药老与萧炎没有师徒关系。 | 药老是萧炎的老师。 | 药老是萧炎的老师。 | PASS | PASS | 是 | 否 | 否 |
| train | `fanjue_identity` | 焚决是什么？ | 焚决是一门能通过吞噬异火来进化的功法。 | 焚决是一门能通过吞噬异火来进化的功法。 | PASS | PASS | 是 | 否 | 否 |
| train | `fanjue_identity` | 请简要说明焚决的性质和特点。 | 焚决是一门能通过吞噬异火来进化的功法。 | 焚决是一门能通过吞噬异火来进化的功法。 | PASS | PASS | 是 | 否 | 否 |
| train | `fanjue_identity` | 焚决属于哪一类修炼法门？ | 焚决是一门能通过吞噬异火来进化的功法。 | 焚决是一门能通过吞噬异火来进化的功法。 | PASS | PASS | 是 | 否 | 否 |
| train | `fanjue_identity` | 焚决靠什么方式进化？ | 焚决是一门能通过吞噬异火来进化的功法。 | 焚决是一门能通过吞噬异火来进化的功法。 | PASS | PASS | 是 | 否 | 否 |
| train | `fanjue_identity` | 焚决不能进化，对吗？请纠正。 | 焚决是一门能通过吞噬异火来进化的功法。 | 焚决是一门能通过吞噬异火来进化的功法。 | PASS | PASS | 是 | 否 | 否 |
| train | `fanjue_identity` | 焚决与异火有什么关系？ | 焚决是一门能通过吞噬异火来进化的功法。 | 焚决是一门能通过吞噬异火来进化的功法。 | PASS | PASS | 是 | 否 | 否 |
| train | `fanjue_identity` | 吞噬异火能让哪门功法进化？ | 焚决是一门能通过吞噬异火来进化的功法。 | 异火可以用于炼药，提高成功率并增强丹药药效。 | FAIL | FAIL | 是 | 否 | 否 |
| train | `fanjue_identity` | 判断正误并说明：焚决的进化与异火无关。 | 焚决是一门能通过吞噬异火来进化的功法。 | 焚决是一门能通过吞噬异火来进化的功法。 | PASS | PASS | 是 | 否 | 否 |
| train | `yihuo_role` | 异火在炼药中有什么作用？ | 异火可以用于炼药，提高成功率并增强丹药药效。 | 异火可以用于炼药，提高成功率并增强丹药药效。 | PASS | PASS | 是 | 否 | 否 |
| train | `yihuo_role` | 炼药时为什么会使用异火？ | 异火可以用于炼药，提高成功率并增强丹药药效。 | 异火可以用于炼药，提高成功率并增强丹药药效。 | PASS | PASS | 是 | 否 | 否 |
| train | `yihuo_role` | 异火能给炼药带来哪些帮助？ | 异火可以用于炼药，提高成功率并增强丹药药效。 | 异火可以用于炼药，提高成功率并增强丹药药效。 | PASS | PASS | 是 | 否 | 否 |
| train | `yihuo_role` | 异火会怎样影响炼药成功率和丹药药效？ | 异火可以用于炼药，提高成功率并增强丹药药效。 | 异火可以用于炼药，提高成功率并增强丹药药效。 | PASS | PASS | 是 | 否 | 否 |
| train | `yihuo_role` | 异火与炼药无关，对吗？请纠正。 | 异火可以用于炼药，提高成功率并增强丹药药效。 | 异火可以用于炼药，提高成功率并增强丹药药效。 | PASS | PASS | 是 | 否 | 否 |
| train | `yihuo_role` | 异火和炼药是什么关系？ | 异火可以用于炼药，提高成功率并增强丹药药效。 | 异火可以用于炼药，提高成功率并增强丹药药效。 | PASS | PASS | 是 | 否 | 否 |
| train | `yihuo_role` | 提高炼药成功率并增强药效的特殊火焰是什么？ | 异火可以用于炼药，提高成功率并增强丹药药效。 | 异火可以用于炼药，提高成功率并增强丹药药效。 | PASS | PASS | 是 | 否 | 否 |
| train | `yihuo_role` | 判断正误并说明：使用异火会降低炼药成功率。 | 异火可以用于炼药，提高成功率并增强丹药药效。 | 异火可以用于炼药，提高成功率并增强丹药药效。 | PASS | PASS | 是 | 否 | 否 |
| train | `yunlanzong_identity` | 云岚宗是什么？ | 云岚宗是加玛帝国境内极有影响力的势力。 | 云岚宗是加玛帝国境内极有影响力的势力。 | PASS | PASS | 是 | 否 | 否 |
| train | `yunlanzong_identity` | 请简要介绍云岚宗。 | 云岚宗是加玛帝国境内极有影响力的势力。 | 云岚宗是加玛帝国境内极有影响力的势力。 | PASS | PASS | 是 | 否 | 否 |
| train | `yunlanzong_identity` | 云岚宗位于哪个帝国境内？ | 云岚宗是加玛帝国境内极有影响力的势力。 | 云岚宗是加玛帝国境内极有影响力的势力。 | PASS | PASS | 是 | 否 | 否 |
| train | `yunlanzong_identity` | 云岚宗在加玛帝国有什么地位？ | 云岚宗是加玛帝国境内极有影响力的势力。 | 云岚宗是加玛帝国境内极有影响力的势力。 | PASS | PASS | 是 | 否 | 否 |
| train | `yunlanzong_identity` | 云岚宗与加玛帝国毫无关系吗？请纠正。 | 云岚宗是加玛帝国境内极有影响力的势力。 | 云岚宗是加玛帝国境内极有影响力的势力。 | PASS | PASS | 是 | 否 | 否 |
| train | `yunlanzong_identity` | 云岚宗和加玛帝国是什么关系？ | 云岚宗是加玛帝国境内极有影响力的势力。 | 云岚宗是加玛帝国境内极有影响力的势力。 | PASS | PASS | 是 | 否 | 否 |
| train | `yunlanzong_identity` | 加玛帝国境内的重要势力之一是什么？ | 云岚宗是加玛帝国境内极有影响力的势力。 | 云岚宗是加玛帝国境内极有影响力的势力。 | PASS | PASS | 是 | 否 | 否 |
| train | `yunlanzong_identity` | 判断正误并说明：云岚宗只是一个普通家庭。 | 云岚宗是加玛帝国境内极有影响力的势力。 | 云岚宗是加玛帝国境内极有影响力的势力。 | PASS | PASS | 是 | 否 | 否 |
| holdout_eval | `xiaoyan_identity` | 小说里的萧炎与萧战是什么亲属关系？ | 萧炎是萧战的儿子。 | 萧战是萧家族长，也是萧炎的父亲。 | FAIL | PASS | 是 | 否 | 否 |
| holdout_eval | `xiaoyan_identity` | 如果有人把萧炎说成萧战的父亲，应如何改正？ | 萧炎是萧战的儿子。 | 萧战是萧战的儿子。 | FAIL | FAIL | 是 | 否 | 是 |
| holdout_eval | `xiaozhan_identity` | 萧家的族长萧战和萧炎有何亲属关系？ | 萧战是萧家族长，也是萧炎的父亲。 | 萧战是萧家族长，也是萧炎的父亲。 | PASS | PASS | 是 | 否 | 否 |
| holdout_eval | `xiaozhan_identity` | 谁既是萧家族长，又是萧炎的父亲？ | 萧战是萧家族长，也是萧炎的父亲。 | 萧战是萧家族长，也是萧炎的父亲。 | PASS | PASS | 是 | 否 | 否 |
| holdout_eval | `yaochen_identity` | 药尘在小说中通常以什么称呼出现？ | 药尘就是药老，也是萧炎的老师。 | 药尘是药老曾经的名字，二者是同一人物。 | FAIL | FAIL | 是 | 否 | 否 |
| holdout_eval | `yaochen_identity` | 药尘对萧炎来说是什么身份？ | 药尘就是药老，也是萧炎的老师。 | 药尘就是药老，也是萧炎的老师。 | PASS | PASS | 是 | 否 | 否 |
| holdout_eval | `yaolao_yaochen_alias` | 药老与药尘是别名关系，还是两名不同人物？ | 药尘是药老曾经的名字，二者是同一人物。 | 药尘是药老曾经的名字，二者是同一人物。 | PASS | PASS | 是 | 否 | 否 |
| holdout_eval | `yaolao_yaochen_alias` | 为什么药老有时会被叫作药尘？ | 药尘是药老曾经的名字，二者是同一人物。 | 药尘就是药老，也是萧炎的老师。 | FAIL | FAIL | 是 | 否 | 否 |
| holdout_eval | `yaolao_teacher` | 在药老和萧炎之间，谁是老师？ | 药老是萧炎的老师。 | 药老是萧炎的老师。 | PASS | PASS | 是 | 否 | 否 |
| holdout_eval | `yaolao_teacher` | 萧炎拜谁为师？ | 药老是萧炎的老师。 | 萧炎是萧战的儿子。 | FAIL | FAIL | 是 | 否 | 否 |
| holdout_eval | `fanjue_identity` | 哪门功法可以借助吞噬异火不断进化？ | 焚决是一门能通过吞噬异火来进化的功法。 | 异火可以用于炼药，提高成功率并增强丹药药效。 | FAIL | FAIL | 是 | 否 | 否 |
| holdout_eval | `fanjue_identity` | 异火对焚决的成长起什么作用？ | 焚决是一门能通过吞噬异火来进化的功法。 | 焚决是一门能通过吞噬异火来进化的功法。 | PASS | PASS | 是 | 否 | 否 |
| holdout_eval | `yihuo_role` | 炼药师使用异火能获得什么好处？ | 异火可以用于炼药，提高成功率并增强丹药药效。 | 异火可以用于炼药，提高成功率并增强丹药药效。 | PASS | PASS | 是 | 否 | 否 |
| holdout_eval | `yihuo_role` | 异火是否能提升成丹概率和丹药效果？ | 异火可以用于炼药，提高成功率并增强丹药药效。 | 异火可以用于炼药，提高成功率并增强丹药药效。 | PASS | PASS | 是 | 否 | 否 |
| holdout_eval | `yunlanzong_identity` | 加玛帝国中的云岚宗属于什么性质的组织？ | 云岚宗是加玛帝国境内极有影响力的势力。 | 云岚宗是加玛帝国境内极有影响力的势力。 | PASS | PASS | 是 | 否 | 否 |
| holdout_eval | `yunlanzong_identity` | 云岚宗在加玛帝国境内是否具有影响力？ | 云岚宗是加玛帝国境内极有影响力的势力。 | 云岚宗是加玛帝国境内极有影响力的势力。 | PASS | PASS | 是 | 否 | 否 |

## 日志与复现

data、generation、validation、checkpoint、orchestrator 分别写入轮转 JSONL。用 `--generation-log-level DEBUG` 或 `GPT_CANARY_LOG_LEVEL_GENERATION=DEBUG` 只调试生成模块；日志不保存问题、答案、生成文本或 Token ID。生产保持 INFO，默认单文件 1 MiB、保留 3 份备份。完整文本仅保存在本报告和对应 JSON 中。
