# Geminator: AI-Powered Code Generation Framework

import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import google.generativeai as genai
import questionary
from dotenv import load_dotenv, set_key
from google.api_core import exceptions as google_exceptions
from rich.console import Console
from rich.syntax import Syntax
from rich.text import Text

# Constants

OUTPUT_DIR = Path("../output")
PROMPTS_DIR = Path("../prompts")
LOG_FILE = Path("../geminator.log")
ENV_FILE = Path("../.env")

ROLE_PROMPTER = "prompter"
ROLE_PROGRAMMER = "programmer"
ROLE_DEBUGGER = "debugger"
ROLE_REFACTORER = "refactorer"
ROLE_EVALUATOR = "evaluator"
AI_ROLES = [
    ROLE_PROMPTER,
    ROLE_PROGRAMMER,
    ROLE_DEBUGGER,
    ROLE_REFACTORER,
    ROLE_EVALUATOR,
]

DEFAULT_MODEL = "gemini-2.5-flash-lite"
DEFAULT_ENV_SETTINGS = {
    "AUTO_MODE": "False",
    **{f"MODEL_{role.upper()}": DEFAULT_MODEL for role in AI_ROLES},
}

# Custom Exceptions


class WorkflowCancelledError(Exception):
    """Workflow was cancelled by the user."""

    pass


# Logger Configuration

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    filename=LOG_FILE,
    filemode="a",
    encoding="utf-8",
)


# UI Components


class TerminalUI:
    """ターミナルUIの描画とユーザー入力を担当するクラス。"""

    def __init__(self):
        self.console = Console()
        self.questionary_style = questionary.Style(
            [
                ("qmark", "fg:#00afff bold"),
                ("question", "bold"),
                ("pointer", "fg:#00afff bold"),
                ("selected", "fg:#ffffff bold bg:#005f87"),
                ("answer", "fg:#00d7ff bold"),
            ]
        )

    def _ask(self, func, message: str, **kwargs) -> Any:
        """
        questionary呼び出しのラッパー。
        KeyboardInterrupt(Ctrl+C)やNone(ESCキー等)が返された場合にWorkflowCancelledErrorを送出する。
        """
        try:
            result = func(
                message, style=self.questionary_style, qmark="?", **kwargs
            ).ask()
            if result is None:
                raise WorkflowCancelledError
            return result
        except KeyboardInterrupt:
            raise WorkflowCancelledError

    def print_header(self):
        ascii_art = """
 .d8888b.  8888888888 888b     d888 8888888 888b    888        d8888 88888888888 .d88888b.  8888888b.
d88P  Y88b 888        8888b   d8888   888   8888b   888       d88888     888    d88P" "Y88b 888   Y88b
888    888 888        88888b.d88888   888   88888b  888      d88P888     888    888     888 888    888
888        8888888    888Y88888P888   888   888Y88b 888     d88P 888     888    888     888 888   d88P
888  88888 888        888 Y888P 888   888   888 Y88b888    d88P  888     888    888     888 8888888P"
888    888 888        888  Y8P  888   888   888  Y88888   d88P   888     888    888     888 888 T88b
Y88b  d88P 888        888   "   888   888   888   Y8888  d8888888888     888    Y88b. .d88P 888  T88b
 "Y8888P88 8888888888 888       888 8888888 888    Y888 d88P     888     888     "Y88888P"  888   T88b
"""
        self.console.print(Text(ascii_art, style="bold cyan", justify="center"))
        self.console.line()

    def print_info(self, message: str):
        self.console.print(f"[bold cyan]>[/bold cyan] {message}")

    def print_success(self, message: str):
        self.console.print(f"[bold green]✔[/bold green] {message}")

    def print_warning(self, message: str):
        self.console.print(f"[bold yellow]⚠[/bold yellow] {message}")

    def print_error(self, message: str, e: Optional[Exception] = None):
        self.console.print(f"[bold red]✖[/bold red] Error: {message}")
        if e:
            logging.error(f"{message}: {e}", exc_info=True)

    def ask_text(self, prompt_text: str, default: Optional[str] = None) -> str:
        return self._ask(questionary.text, prompt_text, default=default or "")

    def ask_password(self, prompt_text: str) -> str:
        return self._ask(questionary.password, prompt_text)

    def ask_choice(self, prompt_text: str, choices: List[Any], **kwargs) -> Any:
        if not choices:
            raise ValueError("ask_choice called with no choices.")
        return self._ask(questionary.select, prompt_text, choices=choices, **kwargs)

    def confirm(self, prompt_text: str, default: bool = False) -> bool:
        return self._ask(questionary.confirm, prompt_text, default=default)

    def display_code(self, code: str, language: str):
        syntax = Syntax(code, language, theme="monokai", line_numbers=True)
        self.console.print(syntax)


# Configuration Management


class ConfigManager:
    """プロジェクト設定と構造を.envファイルで管理するクラス。"""

    def __init__(self, ui: TerminalUI):
        self.ui = ui
        self._ensure_project_structure()
        self._ensure_env_file()
        load_dotenv(ENV_FILE, override=True)

    def _ensure_project_structure(self):
        try:
            OUTPUT_DIR.mkdir(exist_ok=True)
            PROMPTS_DIR.mkdir(exist_ok=True)
        except OSError as e:
            self.ui.print_error("Failed to create necessary directories.", e)
            sys.exit(1)

    def _ensure_env_file(self):
        if not ENV_FILE.exists():
            with ENV_FILE.open("w", encoding="utf-8") as f:
                f.write("# Gemini API Key\n")
                f.write('GEMINI_API_KEY=""\n\n')
                f.write("# Application Settings\n")
                for key, value in DEFAULT_ENV_SETTINGS.items():
                    f.write(f'{key}="{value}"\n')

    def get_value(self, key: str) -> Optional[str]:
        load_dotenv(ENV_FILE, override=True)
        return os.getenv(key)

    def set_value(self, key: str, value: str):
        try:
            set_key(str(ENV_FILE), key, value, quote_mode="always")
            load_dotenv(ENV_FILE, override=True)
            if key != "GEMINI_API_KEY":
                self.ui.print_success(f"{key} updated in .env file.")
        except IOError as e:
            self.ui.print_error(f"Failed to write to .env file.", e)

    def get_api_key(self) -> Optional[str]:
        return self.get_value("GEMINI_API_KEY")

    def get_model(self, role: str) -> str:
        key = f"MODEL_{role.upper()}"
        return self.get_value(key) or DEFAULT_MODEL

    def get_auto_mode(self) -> bool:
        value = self.get_value("AUTO_MODE")
        return value and value.lower() in ["true", "1", "t", "y", "yes"]

    def get_prompt(self, role: str) -> str:
        prompt_file = PROMPTS_DIR / f"{role}.txt"
        try:
            return prompt_file.read_text(encoding="utf-8")
        except FileNotFoundError:
            self.ui.print_error(f"Prompt file not found: '{prompt_file}'")
            self.ui.print_info(
                "Please ensure all prompt files exist in the 'prompts' directory."
            )
            sys.exit(1)
        except IOError as e:
            self.ui.print_error(f"Failed to read prompt file: '{prompt_file}'", e)
            sys.exit(1)


# AI Agent


class AIAgent:
    """Google Gemini APIと対話するAIエージェント。"""

    def __init__(self, role: str, config: ConfigManager, ui: TerminalUI):
        self.role = role
        self.display_name = f"{role.capitalize()} AI"
        self.config = config
        self.ui = ui
        self.model_name = self.config.get_model(self.role)
        self.system_prompt = self.config.get_prompt(self.role)
        self.model = genai.GenerativeModel(
            self.model_name, system_instruction=self.system_prompt
        )

    def generate(self, prompt: str, auto_mode: bool) -> Optional[str]:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                with self.ui.console.status(
                    f"[bold blue]{self.display_name} is thinking...", spinner="dots"
                ):
                    response = self.model.generate_content(prompt)
                    logging.info(f"{self.display_name} response received.")
                    return response.text
            except (
                google_exceptions.GoogleAPICallError,
                google_exceptions.RetryError,
                ValueError,
            ) as e:
                error_msg = (
                    f"{self.display_name} generation failed (Attempt {attempt + 1})"
                )
                self.ui.print_error(error_msg, e)
                if attempt < max_retries - 1:
                    try:
                        if auto_mode or self.ui.confirm(
                            "Retry generation?", default=True
                        ):
                            time.sleep(1)
                            continue
                        else:
                            raise WorkflowCancelledError
                    except WorkflowCancelledError:
                        raise
                self.ui.print_error(
                    f"{self.display_name} generation failed after {max_retries} attempts."
                )
                return None
        return None


# Workflow State


class WorkflowState:

    def __init__(self, project_name: str, language: str, user_instruction: str):
        self.project_name = project_name
        self.language = language
        self.initial_instruction = user_instruction
        self.current_instruction = user_instruction
        self.project_dir = OUTPUT_DIR / project_name
        self.prompter_output: Optional[str] = None
        self.programmer_output: Optional[str] = None
        self.debugger_output: Optional[str] = None
        self.final_code: Optional[str] = None
        self.initial_code_for_enhancement: Optional[str] = None


# Main Application


class Geminator:

    def __init__(self):
        self.ui = TerminalUI()
        self.config_manager = ConfigManager(self.ui)
        self.api_key_configured = False

    @staticmethod
    def _strip_code_block(text: Optional[str]) -> str:
        if not text:
            return ""
        pattern = r"```(?:[a-zA-Z]*)?\n?(.*?)```"
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else text.strip()

    def run(self):
        self.ui.print_header()
        while True:
            try:
                if not self.api_key_configured and not self._setup_api_key():
                    break
                choice = self.ui.ask_choice(
                    "Select an action:",
                    ["New Project", "Enhance Project", "Settings", "Exit"],
                )
                if choice == "New Project":
                    self._run_new_project_workflow()
                elif choice == "Enhance Project":
                    self._run_enhance_workflow()
                elif choice == "Settings":
                    self._show_settings_menu()
                else:  # Exit
                    self.ui.print_info("Exiting Geminator. Goodbye!")
                    break
            except WorkflowCancelledError:
                self.ui.print_warning("Action cancelled. Returning to main menu.")
                # Add a small delay to prevent message overlap with next prompt
                time.sleep(0.1)
            except (KeyboardInterrupt, EOFError):
                print("\n\nInterrupted by user. Exiting.")
                break

    def _setup_api_key(self) -> bool:
        first_attempt = True
        while True:
            api_key = self.config_manager.get_api_key()
            if not api_key:
                if first_attempt:
                    self.ui.print_warning("GEMINI_API_KEY is not set.")
                    first_attempt = False
                api_key_input = self.ui.ask_password(
                    "Please enter your Gemini API Key (or press Enter to exit):"
                )
                if not api_key_input:
                    self.ui.print_error("API Key is required. Exiting.")
                    return False
                self.config_manager.set_value("GEMINI_API_KEY", api_key_input)
                continue

            try:
                genai.configure(api_key=api_key)
                with self.ui.console.status(
                    "[bold blue]Verifying API key...", spinner="dots"
                ):
                    _ = list(genai.list_models())
                self.api_key_configured = True
                self.ui.print_success("Gemini API key is valid and configured successfully.")
                return True
            except google_exceptions.PermissionDenied as e:
                self.ui.print_error(
                    "API key is invalid or has insufficient permissions.", e
                )
                self.config_manager.set_value("GEMINI_API_KEY", "")
                first_attempt = False
            except Exception as e:
                self.ui.print_error("Failed to verify API key.", e)
                self.config_manager.set_value("GEMINI_API_KEY", "")
                first_attempt = False
        return False

    def _run_new_project_workflow(self):
        self.ui.console.rule("[bold green]New Project")
        project_name = self.ui.ask_text("Enter the program name:")
        if not project_name:
            self.ui.print_warning("Program name cannot be empty. Returning to menu.")
            return

        language = self.ui.ask_text(
            "Enter the programming language (e.g., Python):", default="Python"
        )
        if not language:
            self.ui.print_warning("Language cannot be empty. Returning to menu.")
            return

        instruction = self.ui.ask_text("Describe what you want to create:")
        if not instruction:
            self.ui.print_warning("Instruction cannot be empty. Returning to menu.")
            return

        state = WorkflowState(project_name, language, instruction)
        state.project_dir.mkdir(exist_ok=True, parents=True)
        self._execute_full_workflow(state)

    def _run_enhance_workflow(self):
        self.ui.console.rule("[bold green]Enhance Project")
        projects = [p.name for p in OUTPUT_DIR.iterdir() if p.is_dir()]
        if not projects:
            self.ui.print_warning("No projects found in the 'output' directory.")
            return

        project_name = self.ui.ask_choice("Select project to enhance:", projects)
        project_dir = OUTPUT_DIR / project_name
        source_files = sorted(
            [p for p in project_dir.iterdir() if p.is_file()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not source_files:
            self.ui.print_error(f"No source files found for project '{project_name}'.")
            return

        file_to_enhance_path = self.ui.ask_choice(
            "Select file to enhance:", choices=[p.name for p in source_files]
        )
        chosen_file = project_dir / file_to_enhance_path
        try:
            initial_code = chosen_file.read_text(encoding="utf-8")
            language = self._guess_language(chosen_file.suffix)
            self.ui.print_success(f"Loaded '{chosen_file.name}' for enhancement.")
        except IOError as e:
            self.ui.print_error(f"Failed to read file {chosen_file.name}", e)
            return

        instruction = self.ui.ask_text("Describe the enhancements or changes:")
        if not instruction:
            self.ui.print_warning("Enhancement instruction cannot be empty. Returning to menu.")
            return

        state = WorkflowState(project_name, language, instruction)
        state.final_code = initial_code
        state.initial_code_for_enhancement = initial_code
        self._execute_full_workflow(state)

    @staticmethod
    def _guess_language(extension: str) -> str:
        lang_map = {
            ".py": "Python",
            ".js": "JavaScript",
            ".java": "Java",
            ".ts": "TypeScript",
            ".html": "HTML",
        }
        return lang_map.get(extension.lower(), "text")

    def _execute_full_workflow(self, state: WorkflowState):
        auto_mode = self.config_manager.get_auto_mode()
        try:
            agents = {
                role: AIAgent(role, self.config_manager, self.ui) for role in AI_ROLES
            }
        except SystemExit:
            self.ui.print_error("Failed to initialize AI agents. Halting workflow.")
            return

        while True:
            if not self._run_phase(
                state,
                agents[ROLE_PROMPTER],
                self._build_prompter_prompt,
                "prompter_output",
                auto_mode,
            ):
                break

            def run_checked_phase(agent, input_code_attr, output_attr):
                return self._run_generative_phase_with_check(
                    state, agent, input_code_attr, output_attr, auto_mode
                )

            result = run_checked_phase(
                agents[ROLE_PROGRAMMER], "initial_code_for_enhancement", "programmer_output"
            )
            if result == "RESTART":
                self._prepare_for_restart(state, "Programmer AI phase failed.")
                continue
            if not result:
                break

            result = run_checked_phase(
                agents[ROLE_DEBUGGER], "programmer_output", "debugger_output"
            )
            if result == "RESTART":
                self._prepare_for_restart(state, "Debugger AI phase failed.")
                continue
            if not result:
                break

            result = run_checked_phase(
                agents[ROLE_REFACTORER], "debugger_output", "final_code"
            )
            if result == "RESTART":
                self._prepare_for_restart(state, "Refactorer AI phase failed.")
                continue
            if not result:
                break

            self.ui.display_code(state.final_code, state.language)
            self._save_code(state)

            score, title, suggestion = self._run_evaluation_phase(
                state, agents[ROLE_EVALUATOR], auto_mode
            )
            self.ui.print_success(
                f"{agents[ROLE_EVALUATOR].display_name} rated the code: {score}/100"
            )
            if title:
                self.ui.print_info(f"Suggestion Title: [italic]'{title}'[/italic]")

            if auto_mode:
                if score >= 100:
                    self.ui.print_success(
                        "🎉 Program has reached 100 points! Auto-mode finished. 🎉"
                    )
                    break
                state.current_instruction = suggestion or "Improve the code further."
                state.initial_code_for_enhancement = state.final_code
                self.ui.print_info("Automatically enhancing based on suggestion...")
                continue

            if not self._handle_post_workflow(state, suggestion):
                break
            state.initial_code_for_enhancement = state.final_code

    def _run_generative_phase_with_check(
        self,
        state: WorkflowState,
        agent: AIAgent,
        input_attr: str,
        output_attr: str,
        auto_mode: bool,
    ) -> Union[bool, Literal["RESTART"]]:
        attempts = 0
        max_attempts_before_restart = 2

        input_code_for_phase = getattr(state, input_attr) if agent.role != ROLE_PROGRAMMER else state.prompter_output

        while True:
            attempts += 1
            if not self._run_phase(
                state, agent, lambda s: input_code_for_phase, output_attr, auto_mode
            ):
                return False

            new_code = getattr(state, output_attr)
            original_code = (
                getattr(state, input_attr) if agent.role != ROLE_PROGRAMMER else state.initial_code_for_enhancement
            )

            if not (
                original_code and new_code and len(new_code) < len(original_code) * 0.75
            ):
                return True

            warning_msg = (
                f"Generated code by {agent.display_name} is significantly shorter "
                f"({len(new_code)} chars) than the input ({len(original_code)} chars). "
                "This may indicate an error."
            )
            self.ui.print_warning(warning_msg)

            if attempts >= max_attempts_before_restart:
                if auto_mode:
                    self.ui.print_info(
                        "Auto-mode: Restarting workflow due to persistent code reduction issue."
                    )
                    return "RESTART"
                choice = self.ui.ask_choice(
                    "Regeneration also produced short code. What to do?",
                    choices=[
                        "Try to regenerate again",
                        "Restart workflow from the beginning",
                        "Abort workflow",
                    ],
                )
                if choice == "Restart workflow from the beginning":
                    return "RESTART"
                if choice == "Abort workflow":
                    return False
            
            if not auto_mode:
                if not self.ui.confirm("Attempt to regenerate?", default=True):
                    self.ui.print_error(
                        "Aborting due to potentially faulty code generation."
                    )
                    return False
            
            self.ui.print_info(f"Retrying {agent.display_name} generation...")

    def _prepare_for_restart(self, state: WorkflowState, reason: str):
        self.ui.print_warning(f"{reason} Restarting workflow with enhanced instructions.")
        state.current_instruction = (
            "[SYSTEM] A previous attempt failed, possibly due to an error in code generation or evaluation. "
            "Please regenerate the entire script based on the original user instruction, ensuring it is complete, correct, and high-quality. "
            f"\n\nOriginal Instruction: {state.initial_instruction}"
        )
        state.final_code = state.initial_code_for_enhancement
        state.prompter_output = None
        state.programmer_output = None
        state.debugger_output = None

    def _run_phase(self, state, agent, prompt_builder, output_attr, auto_mode):
        prompt = prompt_builder(state)
        if not prompt:
            self.ui.print_error(f"Failed to generate prompt for {agent.display_name}.")
            return False

        try:
            raw_output = agent.generate(prompt, auto_mode)
        except WorkflowCancelledError:
            raise
        if raw_output is None:
            return False

        is_code_agent = agent.role in [
            ROLE_PROGRAMMER,
            ROLE_DEBUGGER,
            ROLE_REFACTORER,
        ]
        processed_output = (
            self._strip_code_block(raw_output) if is_code_agent else raw_output
        )

        if not processed_output:
            self.ui.print_warning(f"{agent.display_name} produced an empty output.")
            
        setattr(state, output_attr, processed_output)
        self.ui.print_success(f"{agent.display_name} has completed its task.")
        logging.info(f"{agent.display_name} Output:\n{processed_output}")
        return True

    @staticmethod
    def _build_prompter_prompt(state: WorkflowState) -> str:
        prompt = f"User Instruction: {state.current_instruction}"
        if state.final_code:
            prompt += f"\n\nCurrent Script:\n```\n{state.final_code}\n```"
        return prompt

    def _run_evaluation_phase(
        self, state: WorkflowState, agent: AIAgent, auto_mode: bool
    ) -> Tuple[int, Optional[str], Optional[str]]:
        prompt = f"User's Original Request: {state.initial_instruction}\n\nFinal Script:\n```\n{state.final_code}\n```"
        while True:
            response = agent.generate(prompt, auto_mode)
            if response is None:
                return 0, None, None
            try:
                cleaned_response = re.search(r"\{.*\}", response, re.DOTALL)
                if not cleaned_response:
                    raise json.JSONDecodeError("No JSON object found", response, 0)
                eval_data = json.loads(cleaned_response.group(0))
                return (
                    int(eval_data.get("score", 0)),
                    eval_data.get("title"),
                    eval_data.get("suggestion"),
                )
            except (json.JSONDecodeError, AttributeError, ValueError) as e:
                self.ui.print_error(f"{agent.display_name} failed to parse response.", e)
                if auto_mode or self.ui.confirm("Retry evaluation?", default=True):
                    time.sleep(1)
                    continue
                return 0, None, None

    def _save_code(self, state: WorkflowState):
        ext_map = {"python": "py", "java": "java", "javascript": "js"}
        ext = ext_map.get(state.language.lower(), "txt")
        version = 1
        while (
            filepath := state.project_dir / f"{state.project_name}_v{version:02d}.{ext}"
        ).exists():
            version += 1
        try:
            if state.final_code:
                filepath.write_text(state.final_code, encoding="utf-8")
                self.ui.print_success(f"Code saved as '{filepath}'")
        except IOError as e:
            self.ui.print_error(f"Failed to save file '{filepath}'", e)

    def _handle_post_workflow(
        self, state: WorkflowState, suggestion: Optional[str]
    ) -> bool:
        choices = ["Modify instructions and enhance", "Finish"]
        if suggestion:
            choices.insert(0, "Enhance based on AI suggestion")

        choice = self.ui.ask_choice("Workflow complete. What's next?", choices)
        if choice == "Enhance based on AI suggestion":
            state.current_instruction = suggestion or ""
            return True
        elif choice == "Modify instructions and enhance":
            new_instruction = self.ui.ask_text(
                "Enter new instructions for enhancement:",
                default=state.current_instruction,
            )
            if new_instruction:
                state.current_instruction = new_instruction
                return True
            else: # Empty input
                self.ui.print_warning("Instruction cannot be empty to continue.")
                return False
        return False

    def _show_settings_menu(self):
        self.ui.console.rule("[bold green]Settings")
        while True:
            choice = self.ui.ask_choice(
                "Select setting to modify:", ["AI Models", "Auto-Mode", "Done"]
            )
            if choice == "AI Models":
                self._modify_model_settings()
            elif choice == "Auto-Mode":
                self._toggle_auto_mode()
            else:  # Done
                break

    def _modify_model_settings(self):
        while True:
            current_models = [
                f"{role.capitalize()} AI: {self.config_manager.get_model(role)}"
                for role in AI_ROLES
            ]
            role_to_change_str = self.ui.ask_choice(
                "Which AI's model to change?", current_models + ["Done"]
            )
            if role_to_change_str == "Done":
                break

            role = role_to_change_str.split(":")[0].replace(" AI", "").lower()
            new_model = self.ui.ask_text(
                f"Enter new model for {role.capitalize()} AI:",
                default=self.config_manager.get_model(role),
            )
            if new_model:
                self.config_manager.set_value(f"MODEL_{role.upper()}", new_model)

    def _toggle_auto_mode(self):
        current_mode = self.config_manager.get_auto_mode()
        message = f"Auto-mode is currently {'ON' if current_mode else 'OFF'}. Turn it "
        new_state = self.ui.confirm(
            message + f"{'OFF' if current_mode else 'ON'}?", default=not current_mode
        )
        if current_mode != new_state:
            self.config_manager.set_value("AUTO_MODE", str(new_state))


def main():
    try:
        app = Geminator()
        app.run()
    except Exception as e:
        logging.critical("An unexpected error occurred in main.", exc_info=True)
        console = Console()
        console.print_exception(show_locals=False)
        console.print(
            f"[bold red]A critical error occurred. Check {LOG_FILE} for details.[/bold red]"
        )


if __name__ == "__main__":
    main()
