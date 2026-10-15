import unittest
from unittest.mock import patch, MagicMock
import subprocess
import base64
from pathlib import Path

from knowledge_mcp.core.git_manager import GitManager

class TestGitManager(unittest.TestCase):
    def setUp(self):
        self.pat_token = "ghp_mocktoken1234567890"
        self.repo_url = "https://github.com/mockuser/mockrepo.git"
        self.dest_path = "/mock/dest/path"

    @patch("subprocess.run")
    def test_clone_https_pat(self, mock_run):
        # Arrange
        mock_run.return_value = MagicMock(stdout="Cloned successfully", returncode=0)
        manager = GitManager(pat_token=self.pat_token)
        
        # Calculate expected base64 basic auth header
        expected_creds = f"x-token-auth:{self.pat_token}"
        expected_b64 = base64.b64encode(expected_creds.encode("utf-8")).decode("utf-8")
        
        # Act
        manager.clone(self.repo_url, self.dest_path)
        
        # Assert
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        
        # Check command structure
        self.assertEqual(cmd[0], "git")
        self.assertEqual(cmd[1], "-c")
        self.assertEqual(cmd[2], f"http.extraHeader=Authorization: Basic {expected_b64}")
        self.assertEqual(cmd[3], "clone")
        self.assertEqual(cmd[4], self.repo_url)
        self.assertEqual(cmd[5], str(Path(self.dest_path)))

    @patch("subprocess.run")
    def test_clone_ssh(self, mock_run):
        # Arrange
        mock_run.return_value = MagicMock(stdout="Cloned successfully", returncode=0)
        
        # We need a mock file to exist for key_path checks
        key_path = "/mock/ssh/id_rsa"
        with patch("pathlib.Path.exists", return_value=True):
            manager = GitManager(ssh_key_path=key_path)
            
            # Act
            manager.clone(self.repo_url, self.dest_path)
            
        # Assert
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        
        self.assertEqual(cmd[0], "git")
        self.assertEqual(cmd[1], "-c")
        self.assertEqual(cmd[2], f'core.sshCommand=ssh -i "{key_path}" -o StrictHostKeyChecking=no')
        self.assertEqual(cmd[3], "clone")
        self.assertEqual(cmd[4], self.repo_url)
        self.assertEqual(cmd[5], str(Path(self.dest_path)))

    @patch("subprocess.run")
    def test_pull_https_pat(self, mock_run):
        # Arrange
        mock_run.return_value = MagicMock(stdout="Pulled successfully", returncode=0)
        manager = GitManager(pat_token=self.pat_token)
        repo_path = "/mock/local/repo"
        
        expected_creds = f"x-token-auth:{self.pat_token}"
        expected_b64 = base64.b64encode(expected_creds.encode("utf-8")).decode("utf-8")
        
        with patch("pathlib.Path.exists", return_value=True):
            # Act
            manager.pull(repo_path, branch="main")
            
        # Assert
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        kwargs = mock_run.call_args[1]
        
        self.assertEqual(cmd[0], "git")
        self.assertEqual(cmd[1], "-c")
        self.assertEqual(cmd[2], f"http.extraHeader=Authorization: Basic {expected_b64}")
        self.assertEqual(cmd[3], "pull")
        self.assertEqual(cmd[4], "origin")
        self.assertEqual(cmd[5], "main")
        self.assertEqual(kwargs.get("cwd"), str(Path(repo_path)))

    @patch("subprocess.run")
    def test_token_masking_on_failure(self, mock_run):
        # Arrange
        expected_creds = f"x-token-auth:{self.pat_token}"
        expected_b64 = base64.b64encode(expected_creds.encode("utf-8")).decode("utf-8")
        
        # Simulate CalledProcessError containing the raw base64 credentials in stderr
        error = subprocess.CalledProcessError(
            returncode=1,
            cmd="git clone",
            stderr=f"fatal: Authentication failed for Basic {expected_b64}"
        )
        mock_run.side_effect = error
        manager = GitManager(pat_token=self.pat_token)
        
        # Act & Assert
        with self.assertRaises(RuntimeError) as context:
            manager.clone(self.repo_url, self.dest_path)
            
        # Verify the exception message masks the base64 token
        self.assertNotIn(expected_b64, str(context.exception))
        self.assertIn("********", str(context.exception))

    @patch("subprocess.run")
    def test_clone_timeout(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="git clone", timeout=180)
        manager = GitManager(pat_token=self.pat_token, timeout=180)
        
        with self.assertRaises(TimeoutError) as ctx:
            manager.clone(self.repo_url, self.dest_path)
            
        self.assertIn("timed out after 180s", str(ctx.exception))

    @patch("subprocess.run")
    def test_pull_timeout(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="git pull", timeout=180)
        manager = GitManager(pat_token=self.pat_token, timeout=180)
        
        with patch("pathlib.Path.exists", return_value=True):
            with self.assertRaises(TimeoutError) as ctx:
                manager.pull("/mock/local/repo", branch="main")
                
        self.assertIn("timed out after 180s", str(ctx.exception))

if __name__ == "__main__":
    unittest.main()

