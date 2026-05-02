"""LongTermMemory 测试

测试用户管理、偏好CRUD、知识CRUD、关键词检索。
"""

from pathlib import Path

import pytest

sys_path = __import__("sys").path
if str(Path(__file__).parent.parent.parent) not in sys_path:
    sys_path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.fixture
def ltm(memory_db_path):
    """创建 LongTermMemory 测试实例"""
    from memory.long_term_memory import LongTermMemory

    return LongTermMemory(memory_db_path)


class TestUserManagement:
    """用户管理测试"""

    def test_get_nonexistent_user_returns_none(self, ltm):
        """测试：不存在用户返回 None"""
        profile = ltm.get_user_profile("no_such_user")
        assert profile is None

    def test_create_user(self, ltm):
        """测试：创建用户成功"""
        ok = ltm.create_or_update_user("user1")
        assert ok

        profile = ltm.get_user_profile("user1")
        assert profile is not None
        assert profile["user_id"] == "user1"
        assert profile["created_at"] is not None
        assert profile["last_active"] is not None

    def test_update_existing_user(self, ltm):
        """测试：更新已有用户活跃时间"""
        ltm.create_or_update_user("user2")
        profile_before = ltm.get_user_profile("user2")

        # 更新用户
        ltm.update_user_activity("user2")
        profile_after = ltm.get_user_profile("user2")

        assert profile_before["user_id"] == profile_after["user_id"]


class TestPreferences:
    """偏好管理测试"""

    def test_save_and_get_preference(self, ltm):
        """测试：保存并读取偏好"""
        ltm.save_preference("user_pref", "favorite_department", "研发部")
        value = ltm.get_preference("user_pref", "favorite_department")
        assert value == "研发部"

    def test_get_nonexistent_preference_returns_default(self, ltm):
        """测试：读取不存在的偏好返回默认值"""
        value = ltm.get_preference("user_no", "key", default="default_val")
        assert value == "default_val"

    def test_get_all_preferences(self, ltm):
        """测试：获取所有偏好"""
        ltm.save_preference("user_all", "key1", "value1")
        ltm.save_preference("user_all", "key2", "value2")

        prefs = ltm.get_all_preferences("user_all")
        assert "key1" in prefs
        assert "key2" in prefs
        assert prefs["key1"] == "value1"

    def test_update_existing_preference(self, ltm):
        """测试：更新已有偏好覆盖旧值"""
        ltm.save_preference("user_upd", "theme", "light")
        ltm.save_preference("user_upd", "theme", "dark")
        value = ltm.get_preference("user_upd", "theme")
        assert value == "dark"

    def test_delete_preference(self, ltm):
        """测试：删除偏好"""
        ltm.save_preference("user_del", "temp_key", "temp")
        ltm.delete_preference("user_del", "temp_key")
        value = ltm.get_preference("user_del", "temp_key")
        assert value is None


class TestKnowledge:
    """知识管理测试"""

    def test_save_and_get_knowledge(self, ltm):
        """测试：保存并检索知识"""
        ltm.save_knowledge(
            "user_k", "常问问题", "经常查询研发部薪资", 0.9
        )
        items = ltm.get_knowledge_by_category("user_k", "常问问题")
        assert len(items) >= 1
        assert "研发部" in items[0]["content"]

    def test_get_relevant_knowledge_by_keyword(self, ltm):
        """测试：关键词检索相关知识"""
        ltm.save_knowledge(
            "user_r", "业务领域", "关注技术部门的薪资分布", 0.9
        )
        ltm.save_knowledge(
            "user_r", "使用习惯", "偏好表格展示", 0.7
        )

        # 用关键词"薪资"检索
        results = ltm.get_relevant_knowledge("user_r", "%薪资%", top_k=3)
        assert len(results) >= 1

    def test_get_all_knowledge_respects_limit(self, ltm):
        """测试：get_all_knowledge 遵守 limit 参数"""
        for i in range(5):
            ltm.save_knowledge("user_lim", f"cat{i % 2}", f"content{i}", 0.5)

        items = ltm.get_all_knowledge("user_lim", limit=3)
        assert len(items) <= 3

    def test_delete_knowledge(self, ltm):
        """测试：删除知识"""
        ltm.save_knowledge("user_delk", "测试", "临时内容", 0.5)
        items = ltm.get_all_knowledge("user_delk", limit=1)
        assert len(items) >= 1

        knowledge_id = items[0]["knowledge_id"]
        ltm.delete_knowledge(knowledge_id)

        items_after = ltm.get_all_knowledge("user_delk", limit=10)
        assert len([i for i in items_after if i["knowledge_id"] == knowledge_id]) == 0
