"""DSPy program builder for creating optimized programs from seed prompts."""

import inspect
from typing import Any, Dict, List, Optional, Type, Union

import dspy
from pydantic import BaseModel

from dspy_bench.config import DSPyConfig
from dspy_bench.data.schema import DatasetExample


class DSPyProgramWrapper(BaseModel):
    """Wrapper for DSPy programs with metadata."""

    program: Any  # dspy.Module instance
    seed_prompt: str
    instruction: Optional[str]
    signature: Optional[str]
    metadata: Dict[str, Any] = dict()


class ProgramBuilder:
    """Builder for creating DSPy programs from seed prompts and configurations."""

    def __init__(self, config: DSPyConfig):
        """Initialize program builder.

        Args:
            config: DSPy configuration
        """
        self.config = config
        self._built_programs: Dict[str, DSPyProgramWrapper] = {}

    def build_program(
        self,
        task_type: str = "classification",
        few_shot_examples: Optional[List[DatasetExample]] = None,
        custom_signature: Optional[Type[dspy.Signature]] = None,
    ) -> DSPyProgramWrapper:
        """Build a DSPy program from the seed prompt.

        Args:
            task_type: Type of task (classification, generation, etc.)
            few_shot_examples: Optional few-shot examples
            custom_signature: Custom DSPy signature class

        Returns:
            Wrapped DSPy program
        """
        # Create or use custom signature
        if custom_signature:
            signature = custom_signature
        else:
            signature = self._create_signature(task_type)

        # Create the program
        program = self._create_program(signature, task_type)

        # Add few-shot examples if provided
        if few_shot_examples:
            program = self._add_few_shot_examples(program, few_shot_examples)

        # Create wrapper
        wrapper = DSPyProgramWrapper(
            program=program,
            seed_prompt=self.config.seed_prompt,
            instruction=self.config.instruction,
            signature=str(signature),
            metadata={
                "task_type": task_type,
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "few_shot_count": len(few_shot_examples) if few_shot_examples else 0,
            }
        )

        # Cache the program
        program_id = f"{task_type}_{len(self._built_programs)}"
        self._built_programs[program_id] = wrapper

        return wrapper

    def _create_signature(self, task_type: str) -> Type[dspy.Signature]:
        """Create a DSPy signature based on task type.

        Args:
            task_type: Type of task

        Returns:
            DSPy signature class
        """
        if task_type == "classification":
            class ClassificationSignature(dspy.Signature):
                """Classify input into appropriate categories."""
                input = dspy.InputField(desc="Input text to classify")
                instruction = dspy.InputField(desc="Classification instruction")
                output = dspy.OutputField(desc="Classification result")

            return ClassificationSignature

        elif task_type == "generation":
            class GenerationSignature(dspy.Signature):
                """Generate text based on input and instruction."""
                input = dspy.InputField(desc="Input text")
                instruction = dspy.InputField(desc="Generation instruction")
                output = dspy.OutputField(desc="Generated text")

            return GenerationSignature

        elif task_type == "qa":
            class QASignature(dspy.Signature):
                """Answer questions based on context."""
                context = dspy.InputField(desc="Context information")
                question = dspy.InputField(desc="Question to answer")
                answer = dspy.OutputField(desc="Answer to the question")

            return QASignature

        elif task_type == "summarization":
            class SummarizationSignature(dspy.Signature):
                """Summarize the given text."""
                text = dspy.InputField(desc="Text to summarize")
                instruction = dspy.InputField(desc="Summarization instruction")
                summary = dspy.OutputField(desc="Summary of the text")

            return SummarizationSignature

        else:
            # Default generic signature
            class GenericSignature(dspy.Signature):
                """Process input according to instruction."""
                input = dspy.InputField(desc="Input text")
                instruction = dspy.InputField(desc="Processing instruction")
                output = dspy.OutputField(desc="Processed output")

            return GenericSignature

    def _create_program(self, signature: Type[dspy.Signature], task_type: str) -> dspy.Module:
        """Create a DSPy program with the given signature.

        Args:
            signature: DSPy signature
            task_type: Task type

        Returns:
            DSPy program/module
        """
        # Format the seed prompt with instruction
        formatted_prompt = self.config.seed_prompt
        if self.config.instruction:
            formatted_prompt = formatted_prompt.replace("{instruction}", self.config.instruction)

        # Create appropriate program based on task type
        if task_type in ["classification", "generation", "qa", "summarization"]:
            # Use Chain of Thought for complex tasks
            with dspy.context(lm=dspy.settings.lm):
                program = dspy.ChainOfThought(signature)
        else:
            # Use simple predict for basic tasks
            with dspy.context(lm=dspy.settings.lm):
                program = dspy.Predict(signature)

        return program

    def _add_few_shot_examples(
        self,
        program: dspy.Module,
        examples: List[DatasetExample],
    ) -> dspy.Module:
        """Add few-shot examples to the program.

        Args:
            program: DSPy program
            examples: Few-shot examples

        Returns:
            Program with few-shot examples
        """
        # Convert DatasetExamples to DSPy examples
        dspy_examples = []
        for example in examples:
            dspy_example = dspy.Example(
                input=example.input,
                output=example.label,
            )
            if hasattr(program, 'signature'):
                # Add instruction if the signature expects it
                if 'instruction' in program.signature.input_fields:
                    dspy_example.with_inputs(instruction=self.config.instruction or "")

            dspy_examples.append(dspy_example)

        # Create a few-shot program
        if hasattr(program, 'with_demonstrations'):
            return program.with_demonstrations(dspy_examples)
        else:
            # For programs that don't support demonstrations directly,
            # we'll need to handle this in the optimizer
            return program

    def get_built_programs(self) -> Dict[str, DSPyProgramWrapper]:
        """Get all built programs.

        Returns:
            Dictionary of built programs
        """
        return self._built_programs.copy()

    def save_program_state(self, program: DSPyProgramWrapper, path: str) -> None:
        """Save program state to disk.

        Args:
            program: Program to save
            path: Save path
        """
        import pickle

        with open(path, "wb") as f:
            pickle.dump(program, f)

    def load_program_state(self, path: str) -> DSPyProgramWrapper:
        """Load program state from disk.

        Args:
            path: Load path

        Returns:
            Loaded program
        """
        import pickle

        with open(path, "rb") as f:
            return pickle.load(f)