import xml.etree.ElementTree as ET
import os
import logging
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from colorama import Fore, Style, init
import time
import uuid
from datetime import datetime

from langchain_deepseek import ChatDeepSeek
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import create_react_agent
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph.message import add_messages

# ------- 环境配置 -------
load_dotenv()
init(autoreset=True)

# ------- 日志配置 -------
def setup_logger():
    """设置日志记录器"""
    logger = logging.getLogger('FinancialAnalysis')
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger

logger = setup_logger()

# ------- 美化输出工具 -------
class OutputFormatter:
    """输出格式化工具"""
    
    @staticmethod
    def header(text: str, level: int = 1) -> str:
        symbols = ["🎯", "📋", "⚙️", "📊", "💡"]
        symbol = symbols[min(level-1, len(symbols)-1)]
        
        if level == 1:
            return f"\n{Fore.CYAN}{Style.BRIGHT}{'='*60}\n{symbol} {text}\n{'='*60}{Style.RESET_ALL}\n"
        elif level == 2:
            return f"\n{Fore.BLUE}{Style.BRIGHT}{symbol} {text}\n{'-'*40}{Style.RESET_ALL}\n"
        else:
            return f"\n{Fore.GREEN}{symbol} {text}{Style.RESET_ALL}\n"
    
    @staticmethod
    def success(text: str) -> str:
        return f"{Fore.GREEN}✅ {text}{Style.RESET_ALL}"
    
    @staticmethod
    def info(text: str) -> str:
        return f"{Fore.BLUE}ℹ️ {text}{Style.RESET_ALL}"

formatter = OutputFormatter()

# 初始化模型
model = ChatDeepSeek(model="deepseek-chat", max_tokens=8000)

# ============= 数据模型定义 =============

class Task(BaseModel):
    id: str = Field(description="任务唯一标识符")
    name: str = Field(description="任务名称")
    description: str = Field(description="任务描述")
    executor_type: str = Field(description="执行器类型")
    dependencies: List[str] = Field(default=[], description="依赖的任务ID")
    parameters: Dict[str, Any] = Field(default={}, description="任务参数")
    status: str = Field(default="pending", description="任务状态")
    result: Optional[str] = Field(default=None, description="任务结果")

class TaskTree(BaseModel):
    root_task: Task
    tasks: List[Task] = Field(description="所有任务列表")
    
    def get_ready_tasks(self) -> List[Task]:
        """获取可执行的任务（依赖已完成）"""
        ready_tasks = []
        for task in self.tasks:
            if task.status == "pending":
                all_deps_completed = all(
                    any(t.id == dep_id and t.status == "completed" for t in self.tasks)
                    for dep_id in task.dependencies
                ) if task.dependencies else True
                
                if all_deps_completed:
                    ready_tasks.append(task)
        return ready_tasks
    
    def update_task_status(self, task_id: str, status: str, result: str = None):
        """更新任务状态"""
        for task in self.tasks:
            if task.id == task_id:
                task.status = status
                if result:
                    task.result = result
                logger.info(f"任务状态更新: {task.name} -> {status}")
                break

# ============= 状态定义 =============

class FinancialAnalysisState(TypedDict):
    messages: Annotated[list, add_messages]
    task_tree: Optional[TaskTree]
    execution_history: List[Dict[str, Any]]
    user_query: str
    workflow_stage: str

# ============= 工具定义 =============

@tool
def get_stock_data(symbol: str, period: str = "1y") -> str:
    """获取股票基础数据"""
    logger.info(f"🔍 收集股票数据: {symbol}, 周期: {period}")
    time.sleep(1)
    
    return f"""
📈 {symbol} 基础数据分析报告
==============================
时间周期: {period}

💰 基础数据:
- 当前价格: HK$125.50
- 总市值: HK$5,000亿
- P/E比率: 18.5
- P/B比率: 2.3
- ROE: 15.2%
- 股息率: 2.1%

📊 价格表现:
- 52周高点: HK$145.20
- 52周低点: HK$98.30
- 日涨跌幅: +2.1%

🔢 财务指标:
- 营收增长率: 8.5% (YoY)
- 净利润率: 22.3%
- 负债率: 45.2%
"""

@tool
def get_financial_news(keyword: str, days: int = 7) -> str:
    """获取金融新闻信息"""
    logger.info(f"📰 分析新闻: {keyword}, 天数: {days}")
    time.sleep(1)
    
    return f"""
📰 {keyword} 新闻情报分析
==============================
时间范围: 最近{days}天

🔥 热点新闻:
1. 【业绩超预期】Q3财报显示营收同比增长15%，净利润增长18%
2. 【重大合作】与多家国际科技公司达成战略合作协议
3. 【股东回报】董事会批准30亿港元股份回购计划
4. 【分析师看好】多家投行上调目标价至HK$150-160

📈 市场情绪:
- 整体情绪: 积极乐观
- 分析师评级: 买入/强力买入 (85%)
- 机构持仓: 持续增加

🎯 关键催化剂:
- 元宇宙业务进展
- 云服务业务增长
- 国际化扩张
"""

@tool
def technical_analysis(symbol: str, indicator: str = "MA") -> str:
    """技术分析工具"""
    logger.info(f"📊 技术分析: {symbol}, 指标: {indicator}")
    time.sleep(1)
    
    return f"""
📊 {symbol} 技术分析报告
==============================
主要指标: {indicator}

📈 移动平均线分析:
- MA5: HK$123.45 (短期支撑)
- MA20: HK$118.20 (中期支撑强劲)
- MA60: HK$115.80 (长期上升趋势)

⚡ 技术指标:
- MACD: 金叉形成，多头信号强烈
- RSI(14): 65 (适度强势，未超买)
- KDJ: K线上穿D线，短线看涨

🎯 关键位分析:
- 强支撑位: HK$120.00, HK$115.00
- 强阻力位: HK$130.00, HK$135.00

💡 技术结论: 多重技术指标显示上涨趋势，建议逢低买入
"""

@tool
def risk_assessment(position_size: str, market_cap: str) -> str:
    """风险评估工具"""
    logger.info(f"🛡️ 评估风险: 仓位={position_size}, 市值={market_cap}")
    time.sleep(1)
    
    return f"""
🛡️ 投资风险评估报告
==============================
持仓规模: {position_size}
市值类型: {market_cap}

⚠️ 风险指标:
- VaR (95%置信度): 日最大损失 2.8%
- Beta系数: 1.15 (略高于市场)
- 最大回撤历史: 35% (2022年)
- 波动率: 28% (年化)

🏗️ 风险因素:
- 监管风险: 中等 (政策变化影响)
- 汇率风险: 低 (主营业务本币)  
- 流动性风险: 极低 (大盘股)

🎯 风险建议:
1. 控制单一持仓比例不超过10%
2. 设置止损位于-15%
3. 定期评估风险敞口
"""

# 搜索工具
search_tool = TavilySearch(max_results=3, topic="general")

# ============= 执行器类 =============

class BaseExecutor:
    """基础执行器"""
    
    def __init__(self, name: str, tools: List[Any]):
        self.name = name
        self.agent = create_react_agent(model, prompt=self._get_prompt(), tools=tools)
    
    def _get_prompt(self) -> str:
        return f"你是{self.name}执行器，专门负责执行特定类型的任务。"
    
    def execute(self, task: Task, context: Dict[str, Any] = None) -> str:
        """执行任务"""
        logger.info(f"⚙️ {self.name}开始执行任务: {task.name}")
        
        task_description = f"""
        任务: {task.name}
        描述: {task.description}
        参数: {task.parameters}
        """
        
        try:
            result = self.agent.invoke({"messages": [HumanMessage(content=task_description)]})
            logger.info(f"✅ {self.name}任务执行成功")
            return result["messages"][-1].content
        except Exception as e:
            logger.error(f"❌ 任务执行失败: {e}")
            return f"执行失败: {str(e)}"

class DataCollectorExecutor(BaseExecutor):
    def __init__(self):
        super().__init__("数据收集器", [get_stock_data, search_tool])
    
    def _get_prompt(self) -> str:
        return "你是专业的金融数据收集器。负责收集股票基础数据、市场数据等信息。"

class TechnicalAnalystExecutor(BaseExecutor):
    def __init__(self):
        super().__init__("技术分析师", [technical_analysis, get_stock_data])
    
    def _get_prompt(self) -> str:
        return "你是专业的技术分析师。负责分析股票的技术指标、图表形态、趋势等。"

class NewsAnalystExecutor(BaseExecutor):
    def __init__(self):
        super().__init__("新闻分析师", [get_financial_news, search_tool])
    
    def _get_prompt(self) -> str:
        return "你是专业的新闻分析师。负责收集和分析金融新闻、市场情绪、行业动态。"

class RiskAssessorExecutor(BaseExecutor):
    def __init__(self):
        super().__init__("风险评估师", [risk_assessment])
    
    def _get_prompt(self) -> str:
        return "你是专业的风险评估师。负责评估投资风险、计算风险指标、提供风险管理建议。"

class ReportGeneratorExecutor(BaseExecutor):
    def __init__(self):
        super().__init__("报告生成器", [])
    
    def _get_prompt(self) -> str:
        return "你是专业的投资分析报告生成器。负责整合各种分析结果，生成完整的投资分析报告。"

# ============= 执行器管理器 =============

class ExecutorManager:
    def __init__(self):
        logger.info("🏗️ 初始化执行器管理器")
        self.executors = {
            "data_collector": DataCollectorExecutor(),
            "technical_analyst": TechnicalAnalystExecutor(), 
            "news_analyst": NewsAnalystExecutor(),
            "risk_assessor": RiskAssessorExecutor(),
            "report_generator": ReportGeneratorExecutor(),
        }
    
    def get_executor(self, executor_type: str) -> BaseExecutor:
        return self.executors.get(executor_type, self.executors["data_collector"])
    
    def execute_task(self, task: Task, context: Dict[str, Any] = None) -> str:
        executor = self.get_executor(task.executor_type)
        return executor.execute(task, context)

# ============= 任务规划器 =============

class TaskPlanner:
    def __init__(self):
        self.chain = (
            ChatPromptTemplate.from_messages([
                ("system", """
                你是专业的金融投资分析任务规划器。根据用户需求，生成分层的任务执行树。
                
                只返回XML格式的任务树：
                <task_tree>
                <root_task>
                <id>root</id>
                <name>主任务名称</name>
                <description>主任务描述</description>
                <executor_type>coordinator</executor_type>
                </root_task>
                <tasks>
                <task>
                <id>task_1</id>
                <name>任务名称</name>
                <description>任务描述</description>
                <executor_type>执行器类型</executor_type>
                <dependencies></dependencies>
                <parameters>
                <symbol>股票代码</symbol>
                </parameters>
                </task>
                </tasks>
                </task_tree>
                """),
                ("human", "{query}")
            ]) | model
        )
    
    def create_task_tree(self, user_query: str) -> TaskTree:
        """根据用户查询创建任务树"""
        logger.info(f"🎯 开始规划任务: {user_query}")
        
        try:
            response = self.chain.invoke({"query": user_query})
            task_tree_xml = response.content
            return self._parse_task_tree_xml(task_tree_xml, user_query)
        except Exception as e:
            logger.error(f"任务树创建失败: {e}")
            return self._create_default_task_tree(user_query)
    
    def _parse_task_tree_xml(self, xml_text: str, user_query: str) -> TaskTree:
        """解析XML格式的任务树"""
        try:
            # 清理XML文本
            xml_text = xml_text.strip()
            if not xml_text.startswith('<'):
                start_idx = xml_text.find('<task_tree>')
                if start_idx != -1:
                    xml_text = xml_text[start_idx:]
                    end_idx = xml_text.find('</task_tree>') + len('</task_tree>')
                    xml_text = xml_text[:end_idx]
            
            root = ET.fromstring(xml_text)
            
            # 解析根任务
            root_task_elem = root.find('root_task')
            root_task = Task(
                id="root",
                name="投资分析",
                description=user_query,
                executor_type="coordinator"
            )
            
            # 解析子任务
            tasks = [root_task]
            tasks_elem = root.find('tasks')
            if tasks_elem is not None:
                for task_elem in tasks_elem.findall('task'):
                    # 解析依赖
                    deps = []
                    deps_elem = task_elem.find('dependencies')
                    if deps_elem is not None and deps_elem.text:
                        deps = [d.strip() for d in deps_elem.text.split(',') if d.strip()]
                    
                    # 解析参数
                    params = {}
                    params_elem = task_elem.find('parameters')
                    if params_elem is not None:
                        for param in params_elem:
                            params[param.tag] = param.text
                    
                    task = Task(
                        id=task_elem.find('id').text or str(uuid.uuid4()),
                        name=task_elem.find('name').text or "默认任务",
                        description=task_elem.find('description').text or "默认描述",
                        executor_type=task_elem.find('executor_type').text or "data_collector",
                        dependencies=deps,
                        parameters=params
                    )
                    tasks.append(task)
            
            return TaskTree(root_task=root_task, tasks=tasks)
            
        except Exception as e:
            logger.error(f"XML解析失败: {e}")
            return self._create_default_task_tree(user_query)
    
    def _create_default_task_tree(self, user_query: str) -> TaskTree:
        """创建默认任务树"""
        logger.warning("使用默认任务树模板")
        
        symbol = "0700.HK"
        if "0700" in user_query or "腾讯" in user_query:
            symbol = "0700.HK"
        
        root_task = Task(
            id="root",
            name="投资分析",
            description=user_query,
            executor_type="coordinator"
        )
        
        tasks = [
            root_task,
            Task(
                id="data_collection",
                name="基础数据收集",
                description=f"收集{symbol}的基础财务和市场数据",
                executor_type="data_collector",
                parameters={"symbol": symbol, "period": "1y"}
            ),
            Task(
                id="technical_analysis",
                name="技术面分析",
                description=f"分析{symbol}的技术指标和图表形态",
                executor_type="technical_analyst",
                dependencies=["data_collection"],
                parameters={"symbol": symbol, "indicators": "MA,RSI,MACD"}
            ),
            Task(
                id="news_analysis",
                name="消息面分析",
                description="分析相关新闻和市场情绪对投资的影响",
                executor_type="news_analyst",
                parameters={"keyword": "腾讯控股", "days": 7}
            ),
            Task(
                id="risk_assessment",
                name="风险评估",
                description="全面评估投资风险和潜在收益",
                executor_type="risk_assessor",
                dependencies=["data_collection"],
                parameters={"position_size": "medium", "market_cap": "large"}
            ),
            Task(
                id="report_generation",
                name="投资报告生成",
                description="整合所有分析结果，生成综合投资建议",
                executor_type="report_generator",
                dependencies=["technical_analysis", "news_analysis", "risk_assessment"]
            )
        ]
        
        return TaskTree(root_task=root_task, tasks=tasks)

# ============= 初始化管理器 =============
task_planner = TaskPlanner()
executor_manager = ExecutorManager()

# ============= 工作流节点定义 =============

def planning_node(state: FinancialAnalysisState) -> Dict[str, Any]:
    """规划节点 - 创建任务树"""
    # 兼容 agent-chat-ui 的消息格式
    user_query = state.get("user_query")
    if not user_query:
        last_message = state["messages"][-1].content
        # 处理结构化消息格式
        if isinstance(last_message, list):
            # 提取文本内容
            text_parts = []
            for part in last_message:
                if isinstance(part, dict) and part.get('type') == 'text':
                    text_parts.append(part.get('text', ''))
            user_query = ' '.join(text_parts)
        else:
            user_query = str(last_message)
    
    logger.info("🎯 进入规划阶段")
    print(formatter.header("🎯 任务规划阶段"))
    print(formatter.info(f"正在分析用户需求: {user_query}"))
    
    # 生成任务树
    task_tree = task_planner.create_task_tree(user_query)
    
    logger.info(f"✅ 任务树创建完成，包含{len(task_tree.tasks)}个任务")
    
    planning_message = f"""
{formatter.header("📋 任务规划完成", 2)}
📊 **规划概览**
- 生成任务总数: {len(task_tree.tasks)}
- 根任务: {task_tree.root_task.name}

💡 **执行策略**
- 系统将自动执行所有任务
- 任务按依赖关系顺序执行

{formatter.success('准备开始执行任务...')}
"""
    
    return {
        "messages": [AIMessage(content=planning_message)],
        "task_tree": task_tree,
        "user_query": user_query,
        "workflow_stage": "planning_complete"
    }

def execution_node(state: FinancialAnalysisState) -> Dict[str, Any]:
    """执行节点 - 自动执行所有任务"""
    task_tree = state.get("task_tree")
    
    if not task_tree:
        return {
            "messages": [AIMessage(content="执行错误：缺少任务树")],
            "workflow_stage": "error"
        }
    
    print(formatter.header("⚙️ 开始执行任务"))
    
    # 按顺序执行所有任务
    while True:
        ready_tasks = task_tree.get_ready_tasks()
        if not ready_tasks:
            break
        
        for task in ready_tasks:
            if task.id == "root":  # 跳过根任务
                task_tree.update_task_status(task.id, "completed")
                continue
            
            print(f"\n{formatter.info(f'正在执行: {task.name}')}")
            
            context = {
                "completed_tasks": [t for t in task_tree.tasks if t.status == "completed"],
                "user_query": state.get("user_query", "")
            }
            
            try:
                result = executor_manager.execute_task(task, context)
                task_tree.update_task_status(task.id, "completed", result)
                print(formatter.success(f"任务完成: {task.name}"))
            except Exception as e:
                logger.error(f"任务执行失败: {e}")
                task_tree.update_task_status(task.id, "failed", str(e))
    
    # 生成最终报告
    completed_tasks = [t for t in task_tree.tasks if t.status == "completed" and t.result]
    
    all_results = []
    for task in completed_tasks:
        if task.id != "root":
            all_results.append(f"**{task.name}**:\n{task.result}\n")
    
    final_report = f"""
{formatter.header("🎉 投资分析报告", 1)}

{formatter.header("📈 分析概览", 2)}
- 分析标的: {state.get('user_query', '投资标的')}
- 完成任务: {len(completed_tasks)} 项
- 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{formatter.header("📊 详细分析结果", 2)}
{"".join(all_results)}

{formatter.header("💡 投资建议", 2)}
基于以上多维度分析，建议投资者：
1. 关注基本面变化和技术信号
2. 注意风险控制和仓位管理  
3. 密切跟踪市场情绪和新闻动态
4. 制定合理的投资策略和时间规划

{formatter.success("分析报告已完成！")}
"""
    
    return {
        "messages": [AIMessage(content=final_report)],
        "workflow_stage": "complete"
    }

# ============= 构建工作流图 =============

def build_analysis_graph():
    """构建简化的分析工作流图"""
    logger.info("🏗️ 构建工作流图")
    
    builder = StateGraph(FinancialAnalysisState)
    
    builder.add_node("planning", planning_node)
    builder.add_node("execution", execution_node)
    
    builder.add_edge(START, "planning")
    builder.add_edge("planning", "execution")
    builder.add_edge("execution", END)
    
    compiled_graph = builder.compile()
    logger.info("✅ 工作流图构建完成")
    
    return compiled_graph
graph = build_analysis_graph()
# "请分析腾讯控股(0700.HK)的投资价值，我想了解其基本面、技术面以及未来6个月的投资前景",
# "分析阿里巴巴的风险收益特征，包括技术分析和新闻情绪", 
# "评估比亚迪股票的投资机会，重点关注基本面和行业前景"


\