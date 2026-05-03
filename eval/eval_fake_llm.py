"""
Mock LLM for evaluation — 比测试 FakeLLM 更精确，按问题内容映射到预设响应。

覆盖所有 26 个 eval cases 所需的 LLM 响应，包括意图识别、SQL 生成、
数据分析、图表配置、结果汇总等场景。
"""

import re
from typing import Dict, Optional, Any


class EvalMockLLM:
    """评估专用 Mock LLM，按问题精确匹配返回预设响应

    匹配优先级：
    1. 意图识别 prompt → 从"当前问题："提取问题，查 intent_map
    2. SQL 生成 prompt → 从"问题："提取问题，查 sql_map
    3. 通用关键词匹配 → 查通用 responses
    """

    def __init__(self):
        self.calls = []

        # 意图识别映射：{问题关键词: intent}
        self.intent_map = {
            "你好": "simple_answer",
            "谢谢": "simple_answer",
            "研发部有多少名员工": "sql_only",
            "研发部有多少人": "sql_only",
            "张三的工资是多少": "sql_only",
            "分析一下刚才": "analysis_only",
            "分析我们公司各部门的薪资水平": "sql_and_analysis",
            "互联网行业软件工程师平均薪资": "web_search",
            "我们公司研发部薪资和行业平均水平": "search_and_sql",
            "公司一共有多少名员工": "sql_only",
            "列出所有部门名称": "sql_only",
            "工资最高的3个人是谁": "sql_only",
            "研发部工资最高的员工叫什么": "sql_only",
            "各部门的平均薪资是多少": "sql_and_analysis",
            "2020年入职的员工有哪些": "sql_only",
            "薪资超过10000的员工人数": "sql_only",
            "北京有哪些部门": "sql_only",
            "公司有几个部门": "sql_only",
            "哪个部门人最多": "sql_only",
            "今天天气": "simple_answer",
            "员工工资排名": "sql_and_analysis",
            "不存在的表": "sql_only",
            "互联网行业趋势": "web_search",
        }

        # SQL 生成映射：{问题关键词: SQL}
        self.sql_map = {
            "公司一共有多少名员工": "SELECT COUNT(*) as total FROM employees",
            "列出所有部门名称": "SELECT dept_name FROM departments ORDER BY dept_name",
            "工资最高的3个人是谁": "SELECT e.emp_name, (s.base_salary + s.bonus) as total_salary FROM employees e JOIN salaries s ON e.emp_id = s.emp_id ORDER BY total_salary DESC LIMIT 3",
            "研发部工资最高的员工叫什么": "SELECT e.emp_name FROM employees e JOIN departments d ON e.dept_id = d.dept_id JOIN salaries s ON e.emp_id = s.emp_id WHERE d.dept_name = '研发部' ORDER BY (s.base_salary + s.bonus) DESC LIMIT 1",
            "各部门的平均薪资是多少": "SELECT d.dept_name, AVG(s.base_salary + s.bonus) as avg_salary FROM departments d JOIN employees e ON d.dept_id = e.dept_id JOIN salaries s ON e.emp_id = s.emp_id GROUP BY d.dept_id, d.dept_name ORDER BY avg_salary DESC",
            "2020年入职的员工有哪些": "SELECT emp_name, position, hire_date FROM employees WHERE hire_date >= '2020-01-01' AND hire_date <= '2020-12-31'",
            "薪资超过10000的员工人数": "SELECT COUNT(*) as count FROM salaries WHERE base_salary + bonus > 10000",
            "北京有哪些部门": "SELECT dept_name FROM departments WHERE location = '北京'",
            "研发部有多少人": "SELECT COUNT(*) as emp_count FROM employees e JOIN departments d ON e.dept_id = d.dept_id WHERE d.dept_name = '研发部'",
            "公司有几个部门": "SELECT COUNT(*) as dept_count FROM departments",
            "哪个部门人最多": "SELECT d.dept_name, COUNT(*) as emp_count FROM departments d JOIN employees e ON d.dept_id = e.dept_id GROUP BY d.dept_id, d.dept_name ORDER BY emp_count DESC LIMIT 1",
            "员工工资排名": "SELECT e.emp_name, d.dept_name, (s.base_salary + s.bonus) as total_salary FROM employees e JOIN departments d ON e.dept_id = d.dept_id JOIN salaries s ON e.emp_id = s.emp_id ORDER BY total_salary DESC",
            "张三的工资": "SELECT e.emp_name, s.base_salary, s.bonus FROM employees e JOIN salaries s ON e.emp_id = s.emp_id WHERE e.emp_name = '张三'",
            "不存在的表": "SELECT * FROM nonexistent_table",
        }

        # 通用响应：{关键词: 响应}
        self.general_map = {
            "数据分析": "## 数据分析报告\n\n### 1. 数据概览\n数据包含多条记录，覆盖各部门的薪资信息。\n\n### 2. 关键发现\n- 研发部平均薪资最高\n- 各部门薪资差异明显\n\n### 3. 建议\n建议定期进行薪资对标分析。",
            "图表配置": '{"title":{"text":"薪资分析"},"tooltip":{},"xAxis":{"data":["研发部","市场部"]},"yAxis":{},"series":[{"type":"bar","data":[18000,15000]}]}',
            "汇总回答": "根据查询结果，{question}的答案是：{data_summary}",
            "纠错": "SELECT e.emp_name FROM employees e JOIN salaries s ON e.emp_id = s.emp_id",
        }

    @staticmethod
    def _extract_question(prompt: str) -> str:
        """从 prompt 中提取用户问题"""
        # 意图识别 prompt
        match = re.search(r'当前问题：(.+?)(?:\n\n|\n请判断)', prompt)
        if match:
            return match.group(1).strip()
        # SQL 生成 prompt — 取最后一条"问题："（避免 Few-shot 示例干扰）
        matches = re.findall(r'问题：(.+)', prompt)
        if matches:
            return matches[-1].strip()
        # 用户问题 prompt（纠错等）
        match = re.search(r'用户问题：(.+?)(?:\n|$)', prompt)
        if match:
            return match.group(1).strip()
        return prompt

    def _match_intent(self, question: str) -> str:
        """匹配意图"""
        if not question or not question.strip():
            return "simple_answer"
        for keyword, intent in self.intent_map.items():
            if keyword in question:
                return intent
        return "simple_answer"  # 默认

    def _match_sql(self, question: str) -> Optional[str]:
        """匹配 SQL 响应"""
        for keyword, sql in self.sql_map.items():
            if keyword in question:
                return sql
        return None

    def _match_general(self, prompt: str) -> Optional[str]:
        """匹配通用响应"""
        for keyword, response in self.general_map.items():
            if keyword in prompt:
                return response
        return None

    def invoke(self, prompt: str):
        """模拟 LLM invoke"""
        question = self._extract_question(prompt)

        # 1. 意图识别 prompt
        if "当前问题：" in prompt:
            intent = self._match_intent(question)
            self.calls.append(("intent", question, intent))
            return _FakeMessage(intent)

        # 2. 意图纠错 prompt
        if "你之前对一个分类任务返回了无效的答案" in prompt:
            intent = self._match_intent(question)
            self.calls.append(("correction", question, intent))
            return _FakeMessage(intent)

        # 3. SQL 生成 prompt
        if "SQL查询专家" in prompt or "请为以下问题生成SQL" in prompt:
            sql = self._match_sql(question)
            if sql is None:
                sql = "SELECT * FROM employees LIMIT 10"
            self.calls.append(("sql_gen", question, sql))
            return _FakeMessage(sql)

        # 4. SQL 纠错 prompt
        if "修复一段出错的SQL" in prompt:
            fixed_sql = self._match_sql(question) or "SELECT e.emp_name FROM employees e"
            self.calls.append(("sql_fix", question, fixed_sql))
            return _FakeMessage(fixed_sql)

        # 5. 分析 prompt
        if "对以下数据进行深度分析" in prompt:
            analysis = self._match_general(prompt) or "## 分析报告\n数据正常。"
            self.calls.append(("analysis", question, analysis[:50]))
            return _FakeMessage(analysis)

        # 6. 图表配置 prompt
        if "生成一个适合可视化的 ECharts" in prompt:
            chart = self.general_map.get("图表配置", "{}")
            self.calls.append(("chart", question, chart[:50]))
            return _FakeMessage(chart)

        # 7. 汇总 prompt
        if "为用户的问题提供一个完整、清晰的回答" in prompt:
            answer = self._build_summary(prompt, question)
            self.calls.append(("summary", question, answer[:50]))
            return _FakeMessage(answer)

        # 8. 总结 prompt
        if "总结以下对话历史" in prompt:
            self.calls.append(("compress", "", "compressed"))
            return _FakeMessage("用户询问了部门和薪资相关的问题。")

        # 9. 记忆提取 prompt
        if "提取用户的偏好信息" in prompt:
            self.calls.append(("memory_pref", "", "{}"))
            return _FakeMessage("{}")
        if "提取值得记住的用户知识点" in prompt:
            self.calls.append(("memory_know", "", "[]"))
            return _FakeMessage("[]")

        # 10. 搜索相关 prompt
        if "联网搜索结果" in prompt or "搜索结果" in prompt:
            answer = "根据行业数据，互联网行业软件工程师平均薪资约为25-35万/年。"
            self.calls.append(("search", question, answer[:50]))
            return _FakeMessage(answer)

        # Default
        self.calls.append(("default", question, "simple_answer"))
        return _FakeMessage("simple_answer")

    def stream(self, prompt: str):
        """模拟 LLM stream"""
        text = self.invoke(prompt).content
        for char in text:
            yield _FakeChunk(char)

    def _build_summary(self, prompt: str, question: str) -> str:
        """构建汇总回答"""
        # 尝试匹配具体的回答
        if "研发部有多少人" in question or "研发部有多少" in prompt:
            return "研发部共有12名员工。"
        if "公司有几个部门" in question:
            return "公司共有12个部门，包括研发部、市场部、人事部、财务部等。"
        if "哪个部门人最多" in question:
            return "研发部人数最多，共有12名员工，其次是销售部有10名员工。"
        if "各部门薪资" in question or "部门薪资情况" in prompt:
            return "各部门薪资分析：\n1. 研发部平均薪资 18,000元\n2. 市场部平均薪资 15,000元\n3. 销售部平均薪资 16,000元\n研发部薪资水平最高。"
        if "员工工资排名" in question:
            return "工资排名前5：\n1. 赵六 - 研发部 - 30,000元\n2. 张三 - 研发部 - 25,000元\n3. 徐峰 - 销售部 - 24,000元\n4. 钱大 - 研发部 - 22,000元\n5. 王五 - 研发部 - 20,000元"
        if "你好" in question or "今天天气" in question:
            return "你好！我是智能数据查询助手，可以帮你查询员工、部门、薪资等信息。有什么可以帮你的吗？"
        return f"根据查询结果，{question}的相关信息已整理完成。"


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChunk:
    def __init__(self, content):
        self.content = content
