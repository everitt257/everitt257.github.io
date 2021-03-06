---
layout: post
author: Xuandong Xu
title: 浅谈数据脱敏
tags: [security, data]
category: work
---

## 背景
写这个主题的主要原因是在南宁开电力行业信息年会时听到了一个关于大数据脱敏算法的实现过程案例。在该会上北邮的博士后为我们介绍了关于大数据脱敏的一些主要测量模型和基于这些模型的难点和痛点而做出来的分布式算法。本人听这个会之前并无对数据脱敏有太多了解，这次借写稿的机会做了一些关于数据脱敏的知识的调查，希望可以起到抛砖引玉的作用。

尽管在大数据平台相对成熟的今天，数据的脱敏依然非常必要。虽然大数据平台一般都会通过用户认证，权限管理以及数据加密这些技术手段来保护敏感数据的隐私，但是对于那些拥有数据权限可以直接或间接接触到这数据的人员来说，譬如ETL工程师或这数据分析家/科学家来说依然存在数据泄漏的风险。与此同时，对于那些没有太多权限访问数据的，同时又存在数据挖掘需求的人员来说，如何合理地给出数据同时提高数据挖掘的价值也是相对较难衡量的一点。我们希望通过数据的脱敏能做到给出的数据拥有充分被挖掘的价值同时保证敏感数据的隐私。

在我们看到后面的介绍之前先来定义一些**术语**[@ashwin_-diversity:_2007]方便理解。

-   属性列（Attribute）:
    可定位到个人的属性，例如街道，姓名，社会安保号/身份证号等。

-   半识别列（Quasi-identifiers）:
    不能直接定位到个人，但是多个这样的属性可以帮助定位到个人的属性。例如生日，年龄，性别，发型颜色等。

-   敏感性列（Sensitive attribute）:
    例如收入，患者疾病，交易数额如上缴电费的信息等。

-   身份泄漏（Identity disclosure）:
    当攻击者通过某些属性确认到这些属性属于某个个体时，为身份泄漏。

-   属性泄漏（Attribute disclosure）:
    当攻击者通过某些属性发现某个个体的新的属性时，称为属性泄漏。一般身份泄漏都会导致属性泄漏，偶尔属性泄漏也会单独发生。

-   相等集（Equivalence class）: 关于多项记录的匿名集合。

看完定义后我们再看一些**假设**来方便理解。

-   半标识列内的信息未必是当前企业/个人独有的，例如关于某个客户的街道信息可能存在于多个数据库。

-   敏感信息未必分布均衡，且敏感信息也分轻重之分。

-   攻击者虽然不能直接获取敏感信息，但是可能了解敏感信息的一些背景，例如某种疾病的发病大致发病机率。

## 数据脱敏方式

需要重新强调的一点，数据脱敏，并不是为了完全隐匿信息，而是在限制隐私数据泄露风险在一定范围内的同时，最大化数据分析挖掘的潜力[@__2015]。
下面所描述的脱敏方式，与其说是一种方式，不如形容其为一种脱敏的思想更加准确。这种思想既可以帮助我们对数据进行脱敏，同时也是一种衡量数据脱敏后数据易泄漏的模型[@__2015]。

### $K$组匿名

$K$组匿名（$K$-Anonymity）。$K$组匿名通过泛化，匿名（泛化的极致）等手段对半标识符进行变形。假设我们只对标识列进行脱敏，可是攻击者任然可能通过公共的关于半标识列的数据库对被攻击者进行定位，从而获取敏感信息。基于这个原因，我们有必要对半标识列进行数据脱敏。这里只举关于数据泛化的例子进行说明。如图1所示，数据泛化对年龄列进行更为泛化的语义替换。

{: style="text-align:center"}
![4组泛化，截图来自Ashwin[@ashwin_-diversity:_2007][]{data-label="fig:KA"}]({{"/data/img/K.PNG" | absolute_url }})

病人的邮编号，年龄，与国籍进行了关于数据泛化的数据变形。所有半识别列相等的行为一个相等集。如\[fig:KA\]
中表二中的第一个相等集行{1,2,3,4}为一个相等集，而行{5,6,7,8}又为一个相等集。相等集种有几行就代表K值为多大的$K$组匿名。如果一张表里所有相等集的$K$值不同，取K值最小的那个称为关于与该表的$K$组匿名。

上面的表1尽管通过K组匿名降低了通过半标识列关联到具体用户的可能性（理论上来说只有$1/K$的概率），但是该模型依然无法保护用户属性列受到攻击。具体来说，根据该模型所展开的属性攻击分两种：同质化属性攻击与背景知识攻击。

-   同质化属性攻击例子：假设攻击者通过其他公开数据源（如社工那里获取），Bob今年31岁，邮编号13068，那么显然Bob存于\[fig:KA\]表二里，攻击者可以通过表二获取得知Bob患有癌症。此时发生属性泄漏。

-   背景知识攻击例子：依然是关于第一条的假设，Alice今年28岁，邮编号13053，若攻击者知晓Alice的某些背景信息，例如Alice患病毒感染的几率非常小这样的时间。那么攻击者也同样可以通过表二获取得知Alice患有心脏病。此时发生属性泄漏。

### $L$多样性

$L$多样性（$L$-Diversity）。$L$多样性在$K$组匿名的基础之上进一步完善。在每个相等集中，敏感属性列的分布至少包含$L$个**恰当的**值来抵御属性泄露[@ashwin_-diversity:_2007]。这句话里面的“恰当”并没有明确定义。一般来说，存在三种关于什么是“$L$个恰当”的定义。为了方便理解，我们用最简单的定义去理解，也就是在敏感属性列中存在至少$L$个不同的值。

作为补充，这里罗列其他两个关于”$L$个恰当“的定义。

-   熵$L$多样性：在一个相等集$E$中定义它的熵为$Entropy(E)$,
    那么当其值大于等于$\log L$时，该相等集的熵符合$L$多样性。其具体定义如下：
    $$\begin{gathered}
    Entropy(E) = -\sum^m_{i}p(E,s_i)\log p(E,s_i) \\
    Entropy(E) \geq \log L\end{gathered}$$
    其中，$p(E,s_i)$表示该相等集中关于该敏感属性分布函数。使用该定义为目去完善的数据变形方法不仅让敏感属性有所不同，且分布均匀。

-   递归$L$多样性：在一个相等集中，定义$r_i$为第$i$个最常见的敏感属性，定义常数$c$。那么我们称其符合递归$L$多样性当以下公式成立：
    $$r_i \leq c(r_l+r_{l+1}+...r_{m-1}+r_{m})$$
    简而言之，如果以这种定义去完善你的数据脱敏算法，那么原本最常出现的敏感属性则不会那么频繁的出现。

以上是三种关于$L$多样性**恰当**的解释。需要注意的一点则是，以上三种解释，并不是具体的关于数据如何脱敏的方法，而是衡量，或者说是数据脱敏的目的。具体到实现其中的任意一种，都离不开数据的变形。这意味着，当我们的衡量方法越复杂，假设越多的时候，对原始数据造成的干扰也越大，数据待挖掘的价值也可能愈发地减小。

以下通过实现第一个定义（敏感属性列存在$L$个不同的值）来举例说明。以之前那个图表为例，我们发现可以挪动与交换一些行数实现$L=3$的多样性。

{: style="text-align:center"}
![3组多样性，截图来自Ashwin[@ashwin_-diversity:_2007][]{data-label="fig:3D"}]({{"/data/img/3D.png" | absolute_url}} )

在这里我们发现，交换一些行数实现了关于3的多样性。如\[fig:3D\]所示，第二个相等集中的行{5,6,7,8}中的{5,6,7}拥有最少为3的不同敏感属性。这意味着发生属性泄露的最低可能性降低至$1/3$。该表符合$L$为3的多样性。

实际上，以上只是一种关于敏感属性分布均匀且相互独立的乐观假设。现实中不一定能如此高效地实现关于$L$的多样性。$L$多样性这个概念，本身也有它的局限性。它局限性体现在以下两点[@ninghui_t-closeness:_2007]:

-   $L$多样性没有实现的必要或者实现难度太大：当表中的敏感数据列得不到相同程度的重视的情况下，实现$L$多样性的收益很低。举某种疾病为例，倘若全球只有$1\%$的人可能获得这种疾病，且那$99\%$的人都对这种疾病公布公布毫不在意，那么我们可以实现$L$多样性的收益非常低。同样的，假设为了实现$L=2$的多样性去对敏感数据进行变形，那么代价将是要改变$99\%$不那么敏感的敏感属性。

-   $L$多样性容易造成“特殊”的属性泄露

    -   相似性攻击：假设表里的敏感属性列有着类似的值或者名字，尽管完成了依据$L$多样性的原则完成了数据脱敏，依然容易造成属性泄露。以下举例说明[@ninghui_t-closeness:_2007]：

        {: style="text-align:center"}
        ![相似性攻击说明[]{data-label="fig:s3"}]({{"/data/img/Simliar3.png" | absolute_url }})

    -   倾斜攻击：假设表里的敏感属性裂存在较大的倾斜，类似某种疾病1:99的分布比例。那么为了满足$L=2$的多样性，我们构建的数据集可能强行将被攻击者的属性曝光率从$1\%$上升至$50\%$。这显然不是我们想看到的现象。

从\[fig:s3\]可以看出，尽管满足了$L=3$的要求，但是假设我们依然知道某个人的半标识符我们仍然可以推断出他大概患了哪种类型的疾病。例如Bob的邮编是476开头的，月薪3K到5K，那么很显然从\[fig:s3\]图中的表4里我们可以得知，或者大概率推断出他患有胃肠道相关的疾病。因此属性泄露仍然存在。

$L$多样性做得一个比较大的假设是攻击者并不知道敏感属性的全局分布概率。但是实际上攻击者往往可以通过数据表推测出这些分布概率大体是怎样的。

### T-closeness

上一个衡量数据脱敏的模型，$L$多样性限制了从$B_0$（未访问任意数据集之前关于某件事情的认知）到$B_2$（访问了脱敏数据集后关于某件事情的认知）的信息增益。实际上在访问$B_2$之前，攻击者会先访问$B_1$（访问了脱离了的半标识列的全局数据集，只包含了敏感数据集的分布）。一般我们无法阻止攻击者访问$B_1$，比如说某种疾病的全球发病率，但是我们可以控制$B_2$也就是脱敏后的数据来限制有关敏感属性列的信息增益。

T-closeness限制了$B_1$至$B_2$关于敏感属性列的信息增益。一个相等集中关于敏感属性的分布与全局的敏感属性的分布所相差的**距离**不超过T，则称这个相等集符合T-closeness。当一个脱敏后的数据表里的所有相等集符合某个T-closeness，则称这个脱敏后的数据表符合T-closeness[@ninghui_t-closeness:_2007]。

T-closeness里面的“距离”概念也是一个相对模糊的概念。如何衡量这个距离也存在分歧，这里限于篇幅不多做展开。比较热门的定义有Variational
Distance，KL
Distance与EMD。另一点需要再次强调，无论是哪一种衡量数据脱敏的模型，都离不开对半标识列的数据变形。为了生成符合多种约束的模型，甚至离不开生成干扰的数据。

## 结论
数据脱敏与数据挖掘两者需要存在一个平衡点。不可能彻底抹除所有用户的标志列进行数据挖掘，那样数据挖掘的价值将大大降低。另一方面，也不可能彻底放开用户的隐私信息，最大化数据挖掘价值，这样会导致用户隐私风险不可控。大数据的脱敏，也不应该完全由算法主导，实际上大数据平台有关脱敏目标还应该包括：基于算法评估脱敏风险的体系，可管理可审批的数据访问机制，以及当数据发生泄露时的可以回溯的审计机制。

本篇的所介绍的数据脱敏方法，基本上都是基于离线的数据脱敏算法。在南宁的会议上所描述的是一种基于内存计算的流处理的数据脱敏实现，在未来这或许是一种数据脱敏的趋势。

{% include mybib.html %}