# xev

一页中文讲解：[jaredpalmer/kev](https://github.com/jaredpalmer/kev) 如何把 Qwen 改造成一个"做选择题"的决策模型，让它给每个选项输出校准过的概率。

A Chinese, plain-language walkthrough of how Kev turns a Qwen model into a decision model that outputs calibrated probabilities over answer options.

## 内容

`index.html` 是一个独立的单页，用浏览器直接打开即可。它用一张客服工单当例子，从头走到尾：

1. 请求如何改写成一串带 special token（`<state>` `<q>` `<opt>` `</opt>` `<decide>`）的序列
2. attention mask（"挡板"）和位置编号如何让每道题互不影响
3. Qwen 只做一次 prefill，不生成文字
4. pointer head 如何用 `<decide>` 与每个 `</opt>` 的 hidden state 打分
5. softmax 和温度校准如何把分数变成概率
6. 概率如何整理成 noul / choice / score 三种答案
7. 训练（交叉熵、数据增强）和温度拟合
8. 与原版 Qwen 的对比：训练和推理各改了什么

页面里有两个可以交互的部分：注意力掩码矩阵，以及一个 pointer head + softmax + 温度的小计算器。

## 来源

- 所有真实数字摘自 jaredpalmer/kev 在 commit [`2074cc4`](https://github.com/jaredpalmer/kev/tree/2074cc44256200bb37540103fa4572901faa10f3) 的 README、模型卡、`PLAN.md` 和源码。
- 页面中引用的少量代码片段来自 kev，按 [Apache-2.0](https://github.com/jaredpalmer/kev/blob/main/LICENSE) 授权，版权归原作者。
- 页面中的"3 个数的笔记"玩具模型、反推的示意分数和部分假设数字是为讲解而编的，页面里都有标注。

本仓库与 Kev、TypeSafe、Qwen 均无关联。
