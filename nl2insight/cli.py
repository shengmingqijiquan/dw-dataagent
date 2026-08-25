"""CLI 入口：python -m nl2insight.cli "需求" [--role data_analyst]"""
import argparse

from nl2insight.agent.graph import run_agent


def main():
    parser = argparse.ArgumentParser(description="nl2insight: NL-to-Insight Agent")
    parser.add_argument("question", help="自然语言需求")
    parser.add_argument("--role", default="data_analyst",
                        choices=["data_analyst", "finance_analyst",
                                 "ops_analyst", "admin"])
    args = parser.parse_args()

    result = run_agent(args.question, args.role)
    print("=" * 60)
    print(result["explanation"])
    print("=" * 60)


if __name__ == "__main__":
    main()
