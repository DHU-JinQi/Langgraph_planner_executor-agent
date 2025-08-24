# Langgraph_planner_executor-agent 项目

一个基于 [LangGraph](https://github.com/langchain-ai/langgraph) 构建的高级工作流系统，实现了复杂的多步骤任务处理和状态管理。

<img width="560" height="816" alt="PE" src="https://github.com/user-attachments/assets/12506a76-efc8-4318-af12-d08a484c1fb8" />

## 架构概述

本项目采用 LangGraph 的图形化工作流设计，由planner规划任务，由executor执行任务。

### 核心组件

- **节点设计**: 每个节点代表一个特定的处理单元，可以是 LLM 调用、工具使用或自定义逻辑
- **条件边**: 支持基于当前状态的动态路由决策

## 视频演示



https://github.com/user-attachments/assets/5f83496f-fd44-4b47-9e44-42e711f2bb76




## 项目结构

```
project/
├── src/                    # 源代码
│   └── agent
│       └── graphs.py            # 工作流图定义
```


## 贡献指南

我们欢迎社区贡献！请参阅 [CONTRIBUTING.md](./CONTRIBUTING.md) 了解如何参与项目开发。

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 支持

如果您遇到问题或有疑问：

1. 查看 [文档](./README.md)
2. 提交 [GitHub Issue](https://github.com/DHU-JinQi/Langgraph_workflow_architecture/issues)



*此项目基于 [LangGraph](https://github.com/langchain-ai/langgraph) 构建，LangGraph 是 LangChain 生态系统的一部分。*
