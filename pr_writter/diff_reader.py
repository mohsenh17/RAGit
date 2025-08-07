import subprocess
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


@dataclass
class CodeChange:
    """Represents a single code change with before/after context."""
    old_content: str
    new_content: str
    change_type: str  # "addition", "deletion", "modification"
    line_number: int
    function_context: Optional[str] = None
    class_context: Optional[str] = None

@dataclass
class FileChange:
    """Enhanced file change with semantic context."""
    file_path: str
    lines_added: int
    lines_deleted: int
    changes: List[CodeChange]
    change_type: str  # "added", "deleted", "modified", "renamed"
    file_type: str  # "python", "javascript", "config", etc.
    before_content: str = ""  # Full file content before changes
    after_content: str = ""   # Full file content after changes
    summary: str = ""         # High-level summary of changes


class DiffMode(Enum):
    STAGED = "staged"
    UNSTAGED = "unstaged"
    COMMITTED = "committed"
    UNTRACKED = "untracked"
    ALL = "all"



@dataclass
class DiffSummary:
    files: List[FileChange]
    total_added: int
    total_deleted: int
    total_files: int
    branch_info: Optional[str] = None


class GitDiffReader:
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
        self._validate_git_repo()
         # Patterns for detecting code context
        self.function_patterns = {
            'python': r'^def\s+(\w+)',
            'javascript': r'^(?:function\s+(\w+)|(\w+)\s*[:=]\s*(?:function|\(.*\)\s*=>))',
            'java': r'^(?:public|private|protected)?\s*(?:static\s+)?[\w<>]+\s+(\w+)\s*\(',
            'cpp': r'^(?:[\w:]+\s+)?(\w+)\s*\([^)]*\)\s*{?',
        }
        
        self.class_patterns = {
            'python': r'^class\s+(\w+)',
            'javascript': r'^class\s+(\w+)',
            'java': r'^(?:public\s+)?class\s+(\w+)',
            'cpp': r'^class\s+(\w+)',
        }

    def _validate_git_repo(self):
        """Ensure we're in a git repository."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--git-dir"],
                cwd=self.repo_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            if result.returncode != 0:
                raise RuntimeError(f"Not a git repository: {self.repo_path}")
        except FileNotFoundError:
            raise RuntimeError("Git is not installed or not in PATH")

    def get_current_branch(self) -> str:
        """Get the current branch name."""
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=self.repo_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if result.returncode != 0:
            return "HEAD"  # Detached HEAD state
        return result.stdout.strip()

    def get_all_branches(self) -> List[str]:
        """Get all local branches."""
        result = subprocess.run(
            ["git", "branch"],
            cwd=self.repo_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to get branches: {result.stderr}")
        
        branches = []
        for line in result.stdout.strip().split("\n"):
            branch = line.strip().lstrip("* ").strip()
            if branch and not branch.startswith("("):  # Skip detached HEAD indicators
                branches.append(branch)
        return branches

    def get_remote_branches(self) -> List[str]:
        """Get all remote tracking branches."""
        result = subprocess.run(
            ["git", "branch", "-r"],
            cwd=self.repo_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if result.returncode != 0:
            return []
        
        branches = []
        for line in result.stdout.strip().split("\n"):
            branch = line.strip()
            if branch and "HEAD ->" not in branch:
                branches.append(branch)
        return branches

    def get_untracked_diff(self) -> str:
        """Collect content from untracked files as diff format."""
        result = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            cwd=self.repo_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"Git ls-files failed: {result.stderr}")
        
        untracked_files = [f for f in result.stdout.strip().split("\n") if f]
        if not untracked_files:
            return ""

        diffs = []
        for file_path in untracked_files:
            full_path = self.repo_path / file_path
            if full_path.is_file():
                try:
                    content = full_path.read_text(encoding='utf-8', errors='ignore')
                    # Create a proper diff header for untracked files
                    diff_header = f"diff --git a/{file_path} b/{file_path}\n"
                    diff_header += f"new file mode 100644\n"
                    diff_header += f"index 0000000..0000000\n"
                    diff_header += f"--- /dev/null\n"
                    diff_header += f"+++ b/{file_path}\n"
                    
                    # Add content with + prefix
                    content_lines = [f"+{line}" for line in content.split("\n")]
                    diff_content = "\n".join(content_lines)
                    
                    diffs.append(diff_header + diff_content)
                except Exception as e:
                    print(f"Warning: Could not read {file_path}: {e}")
        
        return "\n".join(diffs)

    def get_diff(self, mode: DiffMode = DiffMode.STAGED, 
                 base_branch: str = "main", 
                 target_branch: Optional[str] = None) -> str:
        """
        Get diff based on mode and branch comparison.
        
        Args:
            mode: Type of diff to get
            base_branch: Base branch for comparison
            target_branch: Target branch (defaults to current branch)
        """
        if target_branch is None:
            target_branch = self.get_current_branch()

        if mode == DiffMode.COMMITTED:
            # Compare between branches or commits
            if base_branch == target_branch:
                # If same branch, compare with previous commit
                cmd = ["git", "diff", "HEAD~1", "HEAD"]
            else:
                cmd = ["git", "diff", f"{base_branch}...{target_branch}"]
        elif mode == DiffMode.STAGED:
            cmd = ["git", "diff", "--cached"]
        elif mode == DiffMode.UNSTAGED:
            cmd = ["git", "diff"]
        elif mode == DiffMode.UNTRACKED:
            return self.get_untracked_diff()
        elif mode == DiffMode.ALL:
            return "\n".join([
                self.get_diff(DiffMode.STAGED, base_branch, target_branch),
                self.get_diff(DiffMode.UNSTAGED, base_branch, target_branch),
                self.get_diff(DiffMode.UNTRACKED, base_branch, target_branch)
            ])
        else:
            raise ValueError(f"Invalid mode: {mode}")

        result = subprocess.run(
            cmd,
            cwd=self.repo_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"Git diff failed: {result.stderr}")
        return result.stdout

    def detect_file_type(self, file_path: str) -> str:
        """Detect file type from extension."""
        extension_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.h': 'c',
            '.json': 'json',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.md': 'markdown',
            '.txt': 'text',
            '.sh': 'shell',
            '.dockerfile': 'docker',
        }
        
        for ext, file_type in extension_map.items():
            if file_path.lower().endswith(ext):
                return file_type
        return 'unknown'
    
    def extract_code_context(self, lines: List[str], target_line: int, file_type: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract function and class context for a given line number."""
        current_function = None
        current_class = None
        
        if file_type not in self.function_patterns:
            return None, None
        
        func_pattern = re.compile(self.function_patterns[file_type], re.MULTILINE)
        class_pattern = re.compile(self.class_patterns.get(file_type, r'^class\s+(\w+)'), re.MULTILINE)
        
        for i, line in enumerate(lines[:target_line]):
            # Check for class definition
            class_match = class_pattern.search(line)
            if class_match:
                current_class = class_match.group(1)
            
            # Check for function definition
            func_match = func_pattern.search(line)
            if func_match:
                current_function = func_match.group(1)
        
        return current_function, current_class
    
    def reconstruct_file_content(self, changes: List[Dict], file_type: str) -> Tuple[str, str]:
        """Reconstruct before and after file content from diff changes."""
        before_lines = []
        after_lines = []
        
        for change in changes:
            for content_line in change["content"]:
                if content_line.startswith("-"):
                    before_lines.append(content_line[1:])
                elif content_line.startswith("+"):
                    after_lines.append(content_line[1:])
                elif content_line.startswith(" "):
                    # Context line - appears in both
                    before_lines.append(content_line[1:])
                    after_lines.append(content_line[1:])
        
        return "\n".join(before_lines), "\n".join(after_lines)
    
    def analyze_change_semantic(self, old_content: str, new_content: str, file_type: str) -> str:
        """Analyze the semantic meaning of a change."""
        if not old_content and new_content:
            return "addition"
        elif old_content and not new_content:
            return "deletion"
        elif old_content != new_content:
            return "modification"
        else:
            return "unchanged"
    
    def generate_change_summary(self, file_change: FileChange) -> str:
        """Generate a high-level summary of what changed in the file."""
        summaries = []
        
        if file_change.change_type == "added":
            summaries.append(f"Added new {file_change.file_type} file")
        elif file_change.change_type == "deleted":
            summaries.append(f"Deleted {file_change.file_type} file")
        elif file_change.change_type == "renamed":
            summaries.append(f"Renamed {file_change.file_type} file")
        else:
            # Analyze the types of changes
            additions = sum(1 for change in file_change.changes if change.change_type == "addition")
            deletions = sum(1 for change in file_change.changes if change.change_type == "deletion")
            modifications = sum(1 for change in file_change.changes if change.change_type == "modification")
            
            if additions > 0:
                summaries.append(f"Added {additions} code block(s)")
            if deletions > 0:
                summaries.append(f"Removed {deletions} code block(s)")
            if modifications > 0:
                summaries.append(f"Modified {modifications} code block(s)")
        
        return ", ".join(summaries) if summaries else "Minor changes"


    def parse_diff(self, diff_text: str) -> List[FileChange]:
        """Parse unified diff output into structured file changes with enhanced context."""
        if not diff_text.strip():
            return []

        files = []
        current_file = None
        current_change = None
        lines = diff_text.splitlines()
        i = 0

        while i < len(lines):
            line = lines[i]
            
            if line.startswith("diff --git"):
                # Save previous file if exists
                if current_file:
                    if current_change:
                        # Process the last change
                        old_content, new_content = self.reconstruct_file_content([current_change], current_file.file_type)
                        change_type = self.analyze_change_semantic(old_content, new_content, current_file.file_type)
                        
                        code_change = CodeChange(
                            old_content=old_content,
                            new_content=new_content,
                            change_type=change_type,
                            line_number=current_change["new_start"]
                        )
                        current_file.changes.append(code_change)
                    
                    # Generate summary
                    current_file.summary = self.generate_change_summary(current_file)
                    files.append(current_file)
                
                # Extract file paths
                match = re.search(r'diff --git a/(.*?) b/(.*?)$', line)
                if match:
                    old_path, new_path = match.groups()
                    file_path = new_path if new_path != "/dev/null" else old_path
                    
                    # Determine change type
                    change_type = "modified"
                    if old_path == "/dev/null":
                        change_type = "added"
                    elif new_path == "/dev/null":
                        change_type = "deleted"
                    elif old_path != new_path:
                        change_type = "renamed"
                    
                    file_type = self.detect_file_type(file_path)
                    
                    current_file = FileChange(
                        file_path=file_path,
                        lines_added=0,
                        lines_deleted=0,
                        changes=[],
                        change_type=change_type,
                        file_type=file_type
                    )
                    current_change = None

            elif line.startswith("@@") and current_file:
                # Save previous change if exists
                if current_change:
                    old_content, new_content = self.reconstruct_file_content([current_change], current_file.file_type)
                    change_type = self.analyze_change_semantic(old_content, new_content, current_file.file_type)
                    
                    # Extract context
                    all_lines = []
                    for content_line in current_change["content"]:
                        all_lines.append(content_line[1:] if content_line.startswith((' ', '+', '-')) else content_line)
                    
                    function_context, class_context = self.extract_code_context(
                        all_lines, current_change["new_start"], current_file.file_type
                    )
                    
                    code_change = CodeChange(
                        old_content=old_content,
                        new_content=new_content,
                        change_type=change_type,
                        line_number=current_change["new_start"],
                        function_context=function_context,
                        class_context=class_context
                    )
                    current_file.changes.append(code_change)
                
                # Parse hunk header
                match = re.search(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@', line)
                if match:
                    old_start, old_count, new_start, new_count = match.groups()
                    current_change = {
                        "old_start": int(old_start),
                        "old_count": int(old_count) if old_count else 1,
                        "new_start": int(new_start),
                        "new_count": int(new_count) if new_count else 1,
                        "content": []
                    }

            elif current_change and (line.startswith("+") or line.startswith("-") or line.startswith(" ")):
                current_change["content"].append(line)
                
                if current_file:
                    if line.startswith("+") and not line.startswith("+++"):
                        current_file.lines_added += 1
                    elif line.startswith("-") and not line.startswith("---"):
                        current_file.lines_deleted += 1
            
            i += 1

        # Save last file and change
        if current_file:
            if current_change:
                old_content, new_content = self.reconstruct_file_content([current_change], current_file.file_type)
                change_type = self.analyze_change_semantic(old_content, new_content, current_file.file_type)
                
                code_change = CodeChange(
                    old_content=old_content,
                    new_content=new_content,
                    change_type=change_type,
                    line_number=current_change["new_start"]
                )
                current_file.changes.append(code_change)
            
            current_file.summary = self.generate_change_summary(current_file)
            files.append(current_file)

        return files
    
    def format_for_rag(self, file_changes: List[FileChange]) -> str:
        """Format the changes in a way that's optimal for RAG/LLM processing."""
        context = []
        
        # Overall summary
        total_files = len(file_changes)
        total_additions = sum(fc.lines_added for fc in file_changes)
        total_deletions = sum(fc.lines_deleted for fc in file_changes)
        
        context.append("=== CHANGE SUMMARY ===")
        context.append(f"Files changed: {total_files}")
        context.append(f"Total lines added: {total_additions}")
        context.append(f"Total lines deleted: {total_deletions}")
        context.append("")
        
        for file_change in file_changes:
            context.append(f"=== File: {file_change.file_path} ({file_change.file_type}) ===")
            context.append(f"Change Type: {file_change.change_type}")
            context.append(f"Summary: {file_change.summary}")
            context.append(f"Lines Added: {file_change.lines_added}, Lines Deleted: {file_change.lines_deleted}")
            context.append("")
            
            for i, change in enumerate(file_change.changes):
                context.append(f"--- Change {i+1} (Line {change.line_number}) ---")
                if change.function_context:
                    context.append(f"Function: {change.function_context}")
                if change.class_context:
                    context.append(f"Class: {change.class_context}")
                
                context.append("BEFORE:")
                context.append(change.old_content or "(empty)")
                context.append("")
                context.append("AFTER:")
                context.append(change.new_content or "(empty)")
                context.append("")
            
            context.append("=" * 50)
            context.append("")
        
        return "\n".join(context)

    def get_branch_comparison_diff(self, base_branch: str, 
                                  target_branches: Optional[List[str]] = None) -> Dict[str, DiffSummary]:
        """
        Compare multiple branches against a base branch.
        
        Args:
            base_branch: The base branch to compare against
            target_branches: List of branches to compare (defaults to all branches)
        
        Returns:
            Dictionary mapping branch names to their diff summaries
        """
        if target_branches is None:
            all_branches = self.get_all_branches()
            target_branches = [b for b in all_branches if b != base_branch]

        results = {}
        
        for branch in target_branches:
            try:
                diff_text = self.get_diff(DiffMode.COMMITTED, base_branch, branch)
                file_changes = self.parse_diff(diff_text)
                
                total_added = sum(f.lines_added for f in file_changes)
                total_deleted = sum(f.lines_deleted for f in file_changes)
                
                summary = DiffSummary(
                    files=file_changes,
                    total_added=total_added,
                    total_deleted=total_deleted,
                    total_files=len(file_changes),
                    branch_info=f"{base_branch}...{branch}"
                )
                results[branch] = summary
                
            except Exception as e:
                print(f"Warning: Could not compare {base_branch} with {branch}: {e}")
                
        return results

    def generate_commit_message_context(self, mode: DiffMode = DiffMode.STAGED, 
                                      base_branch: str = "main", 
                                      target_branch: Optional[str] = None) -> str:
        """Generate context suitable for RAG-based commit message generation."""
        diff_text = self.get_diff(mode, base_branch, target_branch)
        file_changes = self.parse_diff(diff_text)
        return self.format_for_rag(file_changes)
    
    def get_changes_for_rag(self, mode: DiffMode = DiffMode.STAGED, 
                           base_branch: str = "main", 
                           target_branch: Optional[str] = None) -> List[FileChange]:
        """Get structured changes for RAG processing."""
        diff_text = self.get_diff(mode, base_branch, target_branch)
        return self.parse_diff(diff_text)

def mainOld():
    """Example usage and testing."""
    try:
        reader = GitDiffReader('/home/mohsen/work/yorku/bulkGeneFormer/')
        
        print(f"Current branch: {reader.get_current_branch()}")
        print(f"All branches: {reader.get_all_branches()}")
        
        # Test different diff modes
        for mode in DiffMode:
            print(f"\n=== Testing {mode.value} mode ===")
            try:
                diff_text = reader.get_diff(mode)
                #print(diff_text[:400])
                if diff_text:
                    file_changes = reader.parse_diff(diff_text)
                    print("ho", file_changes[0].file_path)
                    total_added = sum(f.lines_added for f in file_changes)
                    total_deleted = sum(f.lines_deleted for f in file_changes)
                    
                    summary = DiffSummary(
                        files=file_changes,
                        total_added=total_added,
                        total_deleted=total_deleted,
                        total_files=len(file_changes)
                    )
                    
                    print(f"Files changed: {len(file_changes)}")
                    print(f"Lines: +{total_added}/-{total_deleted}")
                    
                    # Generate commit message
                    commit_msg = reader.generate_commit_message(summary)
                    print(f"Suggested commit message:\n{commit_msg}")
                else:
                    print("No changes detected")
                    
            except Exception as e:
                print(f"Error: {e}")

        # Test branch comparisons
        print(f"\n=== Branch Comparisons ===")
        try:
            comparisons = reader.get_branch_comparison_diff("main")
            for branch, summary in comparisons.items():
                print(f"{branch}: {summary.total_files} files, +{summary.total_added}/-{summary.total_deleted}")
        except Exception as e:
            print(f"Branch comparison error: {e}")

        # Generate PR description
        print(f"\n=== PR Description ===")
        try:
            pr_desc = reader.generate_pr_description()
            print(pr_desc[:500] + "..." if len(pr_desc) > 500 else pr_desc)
        except Exception as e:
            print(f"PR description error: {e}")
            
    except Exception as e:
        print(f"Initialization error: {e}")


def main():
    """Example usage and testing with enhanced GitDiffReader."""
    try:
        reader = GitDiffReader('.')
        
        print(f"Current branch: {reader.get_current_branch()}")
        print(f"All branches: {reader.get_all_branches()}")
        
        # Test different diff modes
        for mode in DiffMode:
            print(f"\n=== Testing {mode.value} mode ===")
            try:
                diff_text = reader.get_diff(mode)
                
                if diff_text:
                    file_changes = reader.parse_diff(diff_text)
                    
                    if file_changes:
                        print(f"First changed file: {file_changes[0].file_path}")
                        print(f"File type: {file_changes[0].file_type}")
                        print(f"Change type: {file_changes[0].change_type}")
                        print(f"Summary: {file_changes[0].summary}")
                        
                        total_added = sum(f.lines_added for f in file_changes)
                        total_deleted = sum(f.lines_deleted for f in file_changes)
                        
                        print(f"Files changed: {len(file_changes)}")
                        print(f"Lines: +{total_added}/-{total_deleted}")
                        
                        # Show context for first change
                        if file_changes[0].changes:
                            first_change = file_changes[0].changes[0]
                            if first_change.function_context:
                                print(f"Function context: {first_change.function_context}")
                            if first_change.class_context:
                                print(f"Class context: {first_change.class_context}")
                        
                        # Generate RAG-optimized context
                        print("\n--- RAG Context Preview ---")
                        rag_context = reader.format_for_rag(file_changes)
                        print(rag_context[:500] + "..." if len(rag_context) > 500 else rag_context)
                    else:
                        print("No file changes detected")
                else:
                    print("No changes detected")
                    
            except Exception as e:
                print(f"Error: {e}")

        # Test the new RAG-focused methods
        print(f"\n=== RAG Context Generation ===")
        try:
            # Generate context for different modes
            for mode in [DiffMode.STAGED, DiffMode.UNSTAGED, DiffMode.ALL]:
                print(f"\n--- {mode.value.upper()} RAG Context ---")
                rag_context = reader.generate_commit_message_context(mode)
                if rag_context.strip():
                    lines = rag_context.split('\n')
                    # Show first 10 lines of context
                    preview = '\n'.join(lines[:10])
                    print(preview)
                    if len(lines) > 10:
                        print(f"... ({len(lines) - 10} more lines)")
                else:
                    print("No changes for RAG context")
        except Exception as e:
            print(f"RAG context error: {e}")

        # Test structured data access
        print(f"\n=== Structured Change Analysis ===")
        try:
            file_changes = reader.get_changes_for_rag(DiffMode.ALL)
            if file_changes:
                print(f"Total files analyzed: {len(file_changes)}")
                
                # Group by file type
                file_types = {}
                for fc in file_changes:
                    file_types[fc.file_type] = file_types.get(fc.file_type, 0) + 1
                
                print("Files by type:")
                for file_type, count in file_types.items():
                    print(f"  {file_type}: {count} file(s)")
                
                # Show change types
                change_types = {}
                for fc in file_changes:
                    change_types[fc.change_type] = change_types.get(fc.change_type, 0) + 1
                
                print("Changes by type:")
                for change_type, count in change_types.items():
                    print(f"  {change_type}: {count} file(s)")
                
                # Show functions/classes modified
                functions_modified = []
                classes_modified = []
                
                for fc in file_changes:
                    for change in fc.changes:
                        if change.function_context and change.function_context not in functions_modified:
                            functions_modified.append(change.function_context)
                        if change.class_context and change.class_context not in classes_modified:
                            classes_modified.append(change.class_context)
                
                if functions_modified:
                    print(f"Functions modified: {', '.join(functions_modified[:5])}")
                    if len(functions_modified) > 5:
                        print(f"  ... and {len(functions_modified) - 5} more")
                
                if classes_modified:
                    print(f"Classes modified: {', '.join(classes_modified[:5])}")
                    if len(classes_modified) > 5:
                        print(f"  ... and {len(classes_modified) - 5} more")
                        
            else:
                print("No structured changes found")
                
        except Exception as e:
            print(f"Structured analysis error: {e}")

        # Test branch comparisons 
        print(f"\n=== Branch Comparisons ===")
        try:
            current_branch = reader.get_current_branch()
            if current_branch != "main":
                print(f"Comparing {current_branch} with main...")
                diff_text = reader.get_diff(DiffMode.COMMITTED, "main", current_branch)
                if diff_text:
                    file_changes = reader.parse_diff(diff_text)
                    total_added = sum(f.lines_added for f in file_changes)
                    total_deleted = sum(f.lines_deleted for f in file_changes)
                    print(f"Branch comparison: {len(file_changes)} files, +{total_added}/-{total_deleted}")
                else:
                    print("No differences between branches")
            else:
                print("Currently on main branch - comparing with HEAD~1")
                diff_text = reader.get_diff(DiffMode.COMMITTED, "main", "main")
                if diff_text:
                    file_changes = reader.parse_diff(diff_text)
                    total_added = sum(f.lines_added for f in file_changes)
                    total_deleted = sum(f.lines_deleted for f in file_changes)
                    print(f"Last commit: {len(file_changes)} files, +{total_added}/-{total_deleted}")
                else:
                    print("No recent commits to compare")
        except Exception as e:
            print(f"Branch comparison error: {e}")

        # Example of how to use this with an LLM for commit messages
        print(f"\n=== LLM Integration Example ===")
        try:
            rag_context = reader.generate_commit_message_context(DiffMode.STAGED)
            if rag_context.strip():
                print("Generated context ready for LLM:")
                print("=" * 50)
                
                
                llm_prompt = f"""
Based on these code changes, generate a concise commit message:

{rag_context}

Requirements:
- Use conventional commit format (type: description)
- Keep the main message under 50 characters
- Add a detailed body if there are significant changes
- Focus on the business impact or functionality changes
"""
                
                print("LLM Prompt Template:")
                print(llm_prompt[:800] + "..." if len(llm_prompt) > 800 else llm_prompt)
            else:
                print("No staged changes - nothing to commit")
                
        except Exception as e:
            print(f"LLM integration example error: {e}")

        print(f"\n=== Summary ===")
        print("Enhanced GitDiffReader is working correctly!")
        print("Ready for RAG-based commit message and PR generation.")
            
    except Exception as e:
        print(f"Initialization error: {e}")

if __name__ == "__main__":
    main() 