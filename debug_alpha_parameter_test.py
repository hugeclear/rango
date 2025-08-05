#!/usr/bin/env python3
"""
Debug Alpha Parameter Test
==========================

出力を詳細に確認してパース問題を解決
"""

import subprocess
import time
import json
import re
from pathlib import Path
import shutil


class DebugAlphaTester:
    """デバッグ用α値テスター"""

    def __init__(self):
        self.results = []
        self.debug_outputs = []

    def run_single_test_debug(
        self, alpha_personal: float, alpha_neutral: float
    ) -> dict:
        """単一テストをデバッグモードで実行"""

        print(f"\n🔍 DEBUG: Testing α_p={alpha_personal:.2f}, α_n={alpha_neutral:.2f}")

        try:
            # 評価実行
            start_time = time.time()
            result = subprocess.run(
                ["python", "run_evaluation.py", "--mode", "demo", "--skip-checks"],
                capture_output=True,
                text=True,
                timeout=180,
            )
            end_time = time.time()

            # 出力を保存
            debug_info = {
                "alpha_personal": alpha_personal,
                "alpha_neutral": alpha_neutral,
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "execution_time": end_time - start_time,
            }
            self.debug_outputs.append(debug_info)

            print(f"   Return code: {result.returncode}")
            print(f"   Execution time: {end_time - start_time:.1f}s")

            if result.returncode == 0:
                print("   ✅ Execution successful")

                # STDERRの詳細表示
                print("   📝 STDERR output (last 10 lines):")
                stderr_lines = result.stderr.strip().split("\n")
                for line in stderr_lines[-10:]:
                    print(f"      {line}")

                # STDOUTの詳細表示
                if result.stdout.strip():
                    print("   📝 STDOUT output:")
                    stdout_lines = result.stdout.strip().split("\n")
                    for line in stdout_lines[-5:]:
                        print(f"      {line}")

                # パース試行
                parsed_result = self.parse_output_debug(result.stderr, result.stdout)
                if parsed_result:
                    parsed_result.update(
                        {
                            "alpha_personal": alpha_personal,
                            "alpha_neutral": alpha_neutral,
                            "execution_time": end_time - start_time,
                        }
                    )
                    print(f"   ✅ Parse successful: {parsed_result}")
                    return parsed_result
                else:
                    print("   ❌ Parse failed")
                    self.show_parsing_debug(result.stderr, result.stdout)
                    return None
            else:
                print(f"   ❌ Execution failed")
                print(f"   Error output: {result.stderr[-200:]}")  # 最後の200文字
                return None

        except subprocess.TimeoutExpired:
            print("   ⏰ Timed out")
            return None
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return None

    def parse_output_debug(self, stderr: str, stdout: str) -> dict:
        """デバッグ機能付きパース"""

        # 全出力を結合
        full_output = stderr + "\n" + stdout

        print("   🔍 Parsing attempt...")

        # パフォーマンス数値を探す
        baseline_accuracy = None
        chameleon_accuracy = None
        improvement_rate = None

        # パターン1: 📊 Baseline Performance: の後
        baseline_match = re.search(
            r"📊 Baseline Performance:.*?Accuracy:\s*([0-9.]+)", full_output, re.DOTALL
        )
        if baseline_match:
            baseline_accuracy = float(baseline_match.group(1))
            print(f"      Found baseline accuracy: {baseline_accuracy}")

        # パターン2: 🦎 Chameleon Performance: の後
        chameleon_match = re.search(
            r"🦎 Chameleon Performance:.*?Accuracy:\s*([0-9.]+)", full_output, re.DOTALL
        )
        if chameleon_match:
            chameleon_accuracy = float(chameleon_match.group(1))
            print(f"      Found chameleon accuracy: {chameleon_accuracy}")

        # パターン3: 改善率を直接探す
        improvement_match = re.search(
            r"Improvement Rate:\s*([+-]?[0-9.]+)%", full_output
        )
        if improvement_match:
            improvement_rate = float(improvement_match.group(1)) / 100.0
            print(f"      Found improvement rate: {improvement_rate}")

        # パターン4: より柔軟なAccuracy検索
        if baseline_accuracy is None or chameleon_accuracy is None:
            accuracy_matches = re.findall(r"Accuracy:\s*([0-9.]+)", full_output)
            print(f"      Found accuracy values: {accuracy_matches}")

            if len(accuracy_matches) >= 2:
                if baseline_accuracy is None:
                    baseline_accuracy = float(accuracy_matches[0])
                if chameleon_accuracy is None:
                    chameleon_accuracy = float(accuracy_matches[1])

        # 結果の構築
        if baseline_accuracy is not None and chameleon_accuracy is not None:
            if improvement_rate is None and baseline_accuracy > 0:
                improvement_rate = (
                    chameleon_accuracy - baseline_accuracy
                ) / baseline_accuracy

            result = {
                "baseline_accuracy": baseline_accuracy,
                "chameleon_accuracy": chameleon_accuracy,
                "improvement_rate": improvement_rate or 0,
                "accuracy": chameleon_accuracy,
            }
            print(f"      Parse result: {result}")
            return result
        else:
            print(
                f"      Parse failed: baseline={baseline_accuracy}, chameleon={chameleon_accuracy}"
            )
            return None

    def show_parsing_debug(self, stderr: str, stdout: str):
        """パース失敗時のデバッグ情報表示"""

        full_output = stderr + "\n" + stdout

        print("   🔍 PARSING DEBUG:")

        # Accuracyを含む行を全て表示
        lines = full_output.split("\n")
        accuracy_lines = [
            line for line in lines if "Accuracy" in line or "accuracy" in line
        ]

        if accuracy_lines:
            print("      Lines containing 'Accuracy':")
            for i, line in enumerate(accuracy_lines):
                print(f"        {i+1}: {line.strip()}")
        else:
            print("      No lines containing 'Accuracy' found")

        # Performance関連の行を表示
        performance_lines = [
            line for line in lines if "Performance" in line or "performance" in line
        ]
        if performance_lines:
            print("      Lines containing 'Performance':")
            for i, line in enumerate(performance_lines):
                print(f"        {i+1}: {line.strip()}")

        # 数値を含む行を表示
        numeric_lines = [line for line in lines if re.search(r"\d+\.\d+", line)]
        if numeric_lines:
            print("      Lines containing decimal numbers:")
            for i, line in enumerate(numeric_lines[-10:]):  # 最後の10行
                print(f"        {i+1}: {line.strip()}")

    def test_single_combination(self):
        """単一の組み合わせをテスト（デバッグ用）"""

        print("🦎 Debug Alpha Parameter Test")
        print("=" * 50)

        # 1つだけテスト
        alpha_p, alpha_n = 1.5, -0.8

        result = self.run_single_test_debug(alpha_p, alpha_n)

        if result:
            print(f"\n✅ SUCCESS!")
            print(f"   Improvement: {result['improvement_rate']:+.1%}")
        else:
            print(f"\n❌ FAILED")

        # デバッグ情報を保存
        self.save_debug_info()

    def save_debug_info(self):
        """デバッグ情報を保存"""

        debug_file = "debug_alpha_output.json"
        with open(debug_file, "w") as f:
            json.dump(
                {"timestamp": time.time(), "debug_outputs": self.debug_outputs},
                f,
                indent=2,
            )

        print(f"\n💾 Debug info saved to: {debug_file}")


def main():
    """メイン実行"""
    tester = DebugAlphaTester()
    tester.test_single_combination()


if __name__ == "__main__":
    main()
