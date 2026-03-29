from typing import Dict, Any, List, Optional
import copy
from dataclasses import dataclass
from dm_toolkit.gui.editor.formatters.input_link_formatter import InputLinkFormatter

@dataclass
class InputLinkNode:
    command: Dict[str, Any]
    output_key: Optional[str]
    input_key: Optional[str]
    usage: Optional[str]
    children: List['InputLinkNode']

    @property
    def atype(self) -> str:
        return self.command.get("type", "")

class InputLinkASTBuilder:
    """
    Builds a dependency graph (AST) of commands based on input/output variable linkages.
    Used for robust text generation like 'その数だけ'.
    """

    @classmethod
    def infer_command_labels(cls, commands: List[Any]) -> List[Any]:
        """
        Infers input/output labels for a sequence of commands, avoiding side effects in formatting logic.
        """
        output_label_map: Dict[str, str] = {}
        processed_commands = []

        for command in commands:
            cmd = copy.deepcopy(command) if isinstance(command, dict) else command
            if isinstance(cmd, dict):
                in_key = str(cmd.get("input_value_key") or "")
                saved_label = str(cmd.get("_input_value_label") or "").strip()
                mapped_label = output_label_map.get(in_key, "") if in_key else ""

                # 再発防止: 連鎖コマンドの入力ラベルは generic "クエリ結果" より
                # 推論済みのクエリ/出力ラベルを優先して自然文を生成する。
                if mapped_label and (not saved_label or "クエリ結果" in saved_label or saved_label.startswith("Step ")):
                    cmd["_input_value_label"] = mapped_label

                out_key = str(cmd.get("output_value_key") or "")
                if out_key:
                    inferred_label = InputLinkFormatter.infer_output_value_label(cmd)
                    if inferred_label:
                        output_label_map[out_key] = inferred_label
            processed_commands.append(cmd)

        return processed_commands

    @classmethod
    def build(cls, commands: List[Dict[str, Any]]) -> List[InputLinkNode]:
        nodes = []
        node_map: Dict[str, InputLinkNode] = {}

        for cmd in commands:
            output_key = cmd.get("output_value_key")
            input_key = cmd.get("input_value_key") or cmd.get("input_link")
            usage = cmd.get("input_usage") or cmd.get("input_value_usage")

            node = InputLinkNode(
                command=cmd,
                output_key=output_key,
                input_key=input_key,
                usage=usage,
                children=[]
            )

            if input_key and input_key in node_map:
                # Add this node as a child of the node that produces 'input_key'
                node_map[input_key].children.append(node)
            else:
                # Root node
                nodes.append(node)

            if output_key:
                node_map[output_key] = node

        return nodes

    @classmethod
    def find_producer(cls, commands: List[Dict[str, Any]], input_key: str) -> Optional[InputLinkNode]:
        """Find the AST node that produces the given output key."""
        nodes = cls.build(commands)

        # Helper to search the tree
        def _find_in_tree(node_list: List[InputLinkNode]) -> Optional[InputLinkNode]:
            for node in node_list:
                if node.output_key == input_key:
                    return node
                found = _find_in_tree(node.children)
                if found: return found
            return None

        return _find_in_tree(nodes)
