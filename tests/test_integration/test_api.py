"""REST API 集成测试

测试 Flask API 端点的请求/响应。
"""

import json
from pathlib import Path

import pytest

sys_path = __import__("sys").path
if str(Path(__file__).parent.parent.parent) not in sys_path:
    sys_path.insert(0, str(Path(__file__).parent.parent.parent))


class TestHealthCheck:
    """健康检查"""

    def test_health_returns_200(self, app_client):
        """测试：健康检查返回200"""
        response = app_client.get("/api/health")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["status"] == "healthy"
        assert "features" in data


class TestLogin:
    """登录接口"""

    def test_login_success(self, app_client):
        """测试：登录成功返回用户信息"""
        response = app_client.post(
            "/api/login",
            data=json.dumps({"user_id": "test_user"}),
            content_type="application/json",
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["success"] is True
        assert data["user_id"] == "test_user"
        assert "session_id" in data
        assert "user_info" in data

    def test_login_without_user_id_uses_guest(self, app_client):
        """测试：无user_id时使用guest"""
        response = app_client.post(
            "/api/login",
            data=json.dumps({}),
            content_type="application/json",
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["success"] is True


class TestQuery:
    """查询接口"""

    def test_query_empty_question_returns_400(self, app_client):
        """测试：空问题返回400"""
        # 先登录
        app_client.post(
            "/api/login",
            data=json.dumps({"user_id": "api_test"}),
            content_type="application/json",
        )
        # 空问题
        response = app_client.post(
            "/api/query",
            data=json.dumps({"user_id": "api_test", "question": ""}),
            content_type="application/json",
        )
        assert response.status_code == 400

    def test_query_returns_answer(self, app_client):
        """测试：查询返回回答"""
        # 登录
        app_client.post(
            "/api/login",
            data=json.dumps({"user_id": "api_query"}),
            content_type="application/json",
        )
        # 查询
        response = app_client.post(
            "/api/query",
            data=json.dumps({"user_id": "api_query", "question": "你好"}),
            content_type="application/json",
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["success"] is True
        assert "answer" in data


class TestNewSession:
    """新建会话"""

    def test_new_session_returns_new_id(self, app_client):
        """测试：新建会话返回新 session_id"""
        resp_login = app_client.post(
            "/api/login",
            data=json.dumps({"user_id": "session_test"}),
            content_type="application/json",
        )
        old_session = json.loads(resp_login.data)["session_id"]

        resp_new = app_client.post(
            "/api/new_session",
            data=json.dumps({"user_id": "session_test"}),
            content_type="application/json",
        )
        assert resp_new.status_code == 200
        data = json.loads(resp_new.data)
        assert data["success"] is True
        assert data["session_id"] != old_session


class TestUserInfo:
    """用户信息"""

    def test_user_info_returns_preferences(self, app_client):
        """测试：用户信息包含偏好"""
        app_client.post(
            "/api/login",
            data=json.dumps({"user_id": "info_test"}),
            content_type="application/json",
        )
        resp = app_client.post(
            "/api/user_info",
            data=json.dumps({"user_id": "info_test"}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert data["success"] is True
        assert "user_info" in data
        assert "preferences" in data["user_info"]
        assert "knowledge" in data["user_info"]
