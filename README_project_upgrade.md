# AI Insight Studio v2

这是一个面向求职作品集展示的 Streamlit 数据产品原型，核心目标不是“做几个图”，而是展示：

1. 场景化产品设计能力
2. 数据处理与分析建模能力
3. LLM 与结构化分析结合的能力
4. 能把结果组织成适合业务汇报/面试讲述的面板

## 本版重点升级

### 1) 游戏运营：社媒评论洞察
- 自动选择文本/时间/互动字段
- 评论清洗
- 规则情感分析
- TF-IDF + SVD + KMeans 主题聚类
- Local Outlier Factor 稀有/异常信号发现
- 特殊点标注：高热负面、沉默高关注、困惑信号、竞品比较、故障风险
- 可视化：
  - Semantic Constellation Map
  - Opportunity Matrix
  - Theme Drift Heatstrip
  - 特殊信号表
- LLM 生成汇报摘要

### 2) 商业分析：业务诊断
- 分群优先级矩阵
- 效率前沿图
- Driver Scan（互信息 + 随机森林重要性）
- LLM 输出管理摘要

### 3) 留存分析：用户生命周期
- Cohort Retention Matrix
- Decay Signature
- 生命周期状态划分：新用户 / 连续留存 / 回流 / 流失风险 / 沉默
- 回流机会散点图
- LLM 输出留存建议

## 面试时建议你这样讲
- 我不是做了一个“图表网站”，而是做了一个“场景化 AI 数据分析工作台”
- 核心设计是：先识别数据结构，再走不同业务场景的方法链路
- LLM 不负责替代分析，而是负责把统计、聚类、异常检测结果组织成可汇报叙事

## 部署注意
- 不要在代码里硬编码 API Key
- 在 Streamlit secrets 或环境变量中配置：
  - LLM_API_KEY
  - LLM_BASE_URL（如果走 DeepSeek/OpenAI 兼容接口）
  - LLM_MODEL（可选，默认 deepseek-chat）
