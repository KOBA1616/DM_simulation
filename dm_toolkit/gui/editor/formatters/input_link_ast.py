from typing import Dict, Any, List, Optional
from dataclasses import dataclass

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
