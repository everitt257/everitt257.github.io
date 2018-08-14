---
layout: post
title:  "大数据实验室计划"
category: work
tags: news
comments: true
---
计划如下：

<div class="mermaid">
gantt
    title 大数据实验室计划
    dateFormat YYYY-MM-DD
    
    section 硬件环境       
    机房布置 : done, a1, 2018-04-17, 30d
    安装GPU服务器 : crit, a8, 2019-01-01, 90d
    
    section 软件环境
    购置Tableu第一部分 : done, a2, 2018-04-17, 60d
    购置Tableu第二部分 : crit, buy1, 2019-01-01, 90d
    完善Sophon : active, a3, 2018-05-17, 2018-09-28
    
    section 数据资源
    编制数据资源申请流程 : done, a4, 2018-04-23, 10d
    内部数据资源准备 : done, i1, 2018-04-18, 14d
    部署数据脱敏软件 : active, de2, 2018-07-23, 30d

    section 运营方案
    编制管理规范 : done,a5, 2018-04-18, 15d
    编制操作指引 : done, a6, 2018-04-18, 15d
    编制课题评审标准 : crit, a7, after a6, 5d
    编制积分奖励机制 : crit, a8, after a7, 10d
    拟定数据创新竞赛方案 : crit, a9, after a7, 10d

    section 组织培训
    数据可视化资源准备 : done, l1, 2018-04-18,  7d
    数据可视化入门培训 : done, 2018-05-10 , 2018-05-20
    数据可视化进阶培训 : done, 2018-06-10 , 5d
    数据挖掘入门培训 : done, 2018-07-26, 1d

    section 课题研究
    征集研究课题 : active, k1, 2018-05-15, 180d
    组织专家评审 : crit, k2, 2018-05-20, 180d
    开展课题研究 : active, k3, 2018-05-20, 180d

    section 课题发布
    设计原型: active, x1, 2018-07-25, 30d
    搭建数据创新分享平台 : active, x2, 2018-07-21, 60d
    打造员工知识社区 : x3, 2018-09-30, 60d
    
    section 长远规划
    与高校展开联合认证 : r1, 2018-10-01, 60d
    固话大数据实验室成果 : r2, 2018-10-15, 90d
    制定申报流程 : r3, 2018-10-15, 20d
    支撑竞争性业务 : r4, after r1, 60d
</div>