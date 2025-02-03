# neuroflux/evaluators.py
from datasets import load_dataset
from transformers import GPT2Tokenizer
import torch
import re
import time
import numpy as np
from typing import Dict, List
import math
import torch.nn as nn

class MathReasoningEvaluator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = load_dataset("gsm8k", "main")['test']
    
    def evaluate(self, num_samples=500):
        correct = 0
        for example in self.dataset.shuffle().select(range(num_samples)):
            prompt = f"Q: {example['question']}\nA:"
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").cuda()
            
            with torch.no_grad():
                output = self.model.generate(input_ids, max_length=256)
                answer = self.tokenizer.decode(output[0], skip_special_tokens=True)
            
            if self._extract_answer(answer) == example['answer']:
                correct += 1
        
        return correct / num_samples
    
    def _extract_answer(self, text):
        match = re.search(r"\\boxed{(.+?)}", text)
        return match.group(1).strip() if match else text.split("####")[-1].split("=")[-1].strip()

class NeuroFluxEvaluator:
    """
    Complete benchmark suite from Section 6 of whitepaper
    """
    def __init__(self, model: nn.Module, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
        # Initialize task-specific evaluators
        self.math_evaluator = GSM8KEvaluator(model, tokenizer)
        self.code_evaluator = HumanEvalEvaluator(model, tokenizer)
        self.recovery_evaluator = RecoveryTimeEvaluator(model)
        self.cost_tracker = TrainingCostTracker()
        
    def run_full_benchmark(self) -> Dict:
        """Run complete benchmark suite"""
        results = {}
        
        # Math reasoning (GSM8K)
        results['gsm8k'] = self.math_evaluator.evaluate(num_samples=500)
        
        # Code generation (HumanEval)
        results['humaneval'] = self.code_evaluator.evaluate()
        
        # Recovery time
        results['recovery'] = self.recovery_evaluator.measure_recovery_time()
        
        # Training costs
        results['cost'] = self.cost_tracker.get_cost_metrics()
        
        return results

class GSM8KEvaluator:
    """Math reasoning evaluation on GSM8K dataset"""
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = load_dataset("gsm8k", "main")['test']
        
    def evaluate(self, num_samples: int = 500) -> Dict:
        correct = 0
        total_time = 0
        
        for example in self.dataset.shuffle().select(range(num_samples)):
            start_time = time.time()
            
            # Format prompt
            prompt = f"Q: {example['question']}\nA: Let's solve this step by step:\n"
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").cuda()
            
            # Generate solution
            with torch.no_grad():
                output = self.model.generate(
                    input_ids,
                    max_length=512,
                    temperature=0.7,
                    num_beams=4
                )
                
            # Extract answer
            generated = self.tokenizer.decode(output[0], skip_special_tokens=True)
            predicted = self._extract_answer(generated)
            
            # Check correctness
            if self._normalize_answer(predicted) == self._normalize_answer(example['answer']):
                correct += 1
                
            total_time += time.time() - start_time
            
        return {
            'accuracy': correct / num_samples,
            'throughput': num_samples / total_time,
            'samples': num_samples
        }
        
    def _extract_answer(self, text: str) -> str:
        """Extract final answer from generated solution"""
        # Look for answer in ####
        if "####" in text:
            return text.split("####")[-1].strip()
        # Look for boxed answer
        match = re.search(r"\\boxed{(.+?)}", text)
        if match:
            return match.group(1).strip()
        # Default to last number in text
        numbers = re.findall(r'-?\d*\.?\d+', text)
        return numbers[-1] if numbers else ""
        
    def _normalize_answer(self, answer: str) -> float:
        """Normalize answer for comparison"""
        try:
            return float(re.findall(r'-?\d*\.?\d+', str(answer))[0])
        except:
            return float('nan')

class HumanEvalEvaluator:
    """Code generation evaluation on HumanEval benchmark"""
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = load_dataset("openai_humaneval")['test']
        
    def evaluate(self, num_samples: int = 100) -> Dict:
        results = {
            'pass@1': 0,
            'pass@10': 0,
            'samples': num_samples
        }
        
        for example in self.dataset.select(range(num_samples)):
            # Generate multiple samples for pass@k
            samples = self._generate_samples(example['prompt'], k=10)
            
            # Test each sample
            passed = [
                self._test_solution(example['test'], sample)
                for sample in samples
            ]
            
            # Update metrics
            results['pass@1'] += passed[0]
            results['pass@10'] += any(passed)
            
        # Normalize results
        results['pass@1'] /= num_samples
        results['pass@10'] /= num_samples
        
        return results
        
    def _generate_samples(self, prompt: str, k: int = 10) -> List[str]:
        """Generate k different completions"""
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").cuda()
        
        outputs = []
        for _ in range(k):
            with torch.no_grad():
                output = self.model.generate(
                    input_ids,
                    max_length=512,
                    temperature=0.8,
                    top_p=0.95,
                    do_sample=True
                )
                outputs.append(
                    self.tokenizer.decode(output[0], skip_special_tokens=True)
                )
                
        return outputs
        
    def _test_solution(self, test_case: str, solution: str) -> bool:
        """Test if solution passes test case"""
        try:
            namespace = {}
            exec(solution, namespace)
            exec(test_case, namespace)
            return True
        except:
            return False

class RecoveryTimeEvaluator:
    """Measure system recovery time after failures"""
    def __init__(self, model):
        self.model = model
        
    def measure_recovery_time(self, num_trials: int = 10) -> Dict:
        recovery_times = []
        
        for _ in range(num_trials):
            # Simulate failure
            self._simulate_failure()
            
            # Measure recovery
            start_time = time.time()
            self._recover()
            recovery_time = time.time() - start_time
            
            recovery_times.append(recovery_time)
            
        return {
            'mean_recovery_time': np.mean(recovery_times),
            'std_recovery_time': np.std(recovery_times),
            'trials': num_trials
        }
        
    def _simulate_failure(self):
        """Simulate random GPU/node failures"""
        # Randomly corrupt model states
        for param in self.model.parameters():
            if torch.rand(1).item() < 0.1:  # 10% failure rate
                param.data.mul_(torch.randn_like(param.data))
                
    def _recover(self):
        """Recover from simulated failure"""
        # Attempt RAID recovery
        self.model.raid.recover_from_failure()
        
        # Verify model state
        self.model.verify_state()

class TrainingCostTracker:
    """Track training costs and resource usage"""
    def __init__(self):
        self.start_time = time.time()
        self.gpu_hours = 0
        self.total_tokens = 0
        
    def update(self, batch_size: int, seq_length: int):
        """Update metrics after processing batch"""
        self.total_tokens += batch_size * seq_length
        
        # Update GPU hours (assuming all GPUs active)
        current_time = time.time()
        self.gpu_hours += (
            (current_time - self.start_time) / 3600 * 
            torch.cuda.device_count()
        )
        self.start_time = current_time
        
    def get_cost_metrics(self) -> Dict:
        """Get current cost metrics"""
        # Assuming $0.5 per GPU hour (typical cloud cost)
        gpu_cost = self.gpu_hours * 0.5
        
        return {
            'gpu_hours': self.gpu_hours,
            'total_tokens': self.total_tokens,
            'tokens_per_gpu_hour': self.total_tokens / max(1, self.gpu_hours),
            'estimated_cost': gpu_cost,
            'cost_per_million_tokens': (gpu_cost * 1e6) / max(1, self.total_tokens)
        }

class LoRALayer(nn.Module):
    """
    Low-Rank Adaptation layer for XLSTM fine-tuning
    Used during exploitation phase
    """
    def __init__(
        self,
        base_layer: nn.Module,
        rank: int = 8,
        alpha: float = 16
    ):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        
        # Get base layer dimensions
        if isinstance(base_layer, nn.Linear):
            in_features = base_layer.in_features
            out_features = base_layer.out_features
        elif isinstance(base_layer, nn.LSTMCell):
            in_features = base_layer.input_size
            out_features = base_layer.hidden_size * 4  # LSTM has 4 gates
            
        # Initialize LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        
        # Initialize with small random values
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
        # Scaling factor
        self.scaling = alpha / rank
        
    def forward(self, x: torch.Tensor, *args, **kwargs):
        """Forward pass with LoRA adaptation"""
        # Base layer computation
        base_output = self.base_layer(x, *args, **kwargs)
        
        # LoRA adaptation
        if isinstance(self.base_layer, nn.Linear):
            lora_output = (x @ self.lora_A @ self.lora_B) * self.scaling
            return base_output + lora_output
        elif isinstance(self.base_layer, nn.LSTMCell):
            # For LSTM, adapt the input-to-hidden transformation
            lora_output = (x @ self.lora_A @ self.lora_B) * self.scaling
            lora_output = lora_output.view(*base_output[0].shape)
            return (base_output[0] + lora_output, base_output[1])
            
    def regularization_loss(self) -> torch.Tensor:
        """L2 regularization for LoRA parameters"""
        return (
            torch.norm(self.lora_A) ** 2 +
            torch.norm(self.lora_B) ** 2
        ) * 0.5

def add_lora_to_model(model: nn.Module) -> None:
    """Add LoRA adapters to XLSTM components"""
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.LSTMCell)):
            if 'xlstm' in name:  # Only adapt XLSTM components
                setattr(
                    model,
                    name,
                    LoRALayer(module)
                )

class CodeGenerationEvaluator:
    """
    Code generation evaluator supporting HumanEval and MBPP benchmarks
    Implements evaluation protocol from Section 6.2 of whitepaper
    """
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        dataset: str = "humaneval",
        num_samples: int = 200,
        timeout: int = 5
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = load_dataset(dataset)['test']
        self.num_samples = num_samples
        self.timeout = timeout
        
        # Test execution environment setup
        self.execution_env = ExecutionEnvironment(timeout=timeout)
        
        # Metrics tracking
        self.metrics = {
            'pass@1': 0.0,
            'pass@10': 0.0,
            'pass@100': 0.0,
            'syntax_validity': 0.0,
            'execution_time': 0.0,
            'memory_usage': 0.0
        }
        
    def evaluate(self, temperature: float = 0.8) -> Dict[str, float]:
        """
        Run complete evaluation suite with detailed metrics
        """
        total_samples = len(self.dataset)
        syntax_valid = 0
        total_exec_time = 0
        total_memory = 0
        
        # Track passing solutions for pass@k
        passing_solutions = []
        
        for idx, example in enumerate(self.dataset):
            print(f"Evaluating example {idx+1}/{total_samples}")
            
            # Generate multiple solutions
            solutions = self._generate_solutions(
                example['prompt'],
                n_samples=self.num_samples,
                temperature=temperature
            )
            
            # Test solutions
            results = self._test_solutions(solutions, example['test'])
            
            # Track metrics
            syntax_valid += sum(1 for r in results if r['syntax_valid'])
            total_exec_time += sum(r['execution_time'] for r in results)
            total_memory += sum(r['memory_usage'] for r in results)
            
            # Track passing solutions for pass@k
            passing = [r['passed'] for r in results]
            passing_solutions.append(passing)
            
        # Compute pass@k metrics
        self.metrics['pass@1'] = self._compute_pass_k(passing_solutions, k=1)
        self.metrics['pass@10'] = self._compute_pass_k(passing_solutions, k=10)
        self.metrics['pass@100'] = self._compute_pass_k(passing_solutions, k=100)
        
        # Compute other metrics
        self.metrics['syntax_validity'] = syntax_valid / (total_samples * self.num_samples)
        self.metrics['execution_time'] = total_exec_time / (total_samples * self.num_samples)
        self.metrics['memory_usage'] = total_memory / (total_samples * self.num_samples)
        
        return self.metrics
    
    def _generate_solutions(
        self,
        prompt: str,
        n_samples: int,
        temperature: float
    ) -> List[str]:
        """Generate multiple solutions for a given prompt"""
        solutions = []
        
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").cuda()
        
        # Generate solutions
        for _ in range(n_samples):
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    max_length=512,
                    temperature=temperature,
                    top_p=0.95,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
                
            # Decode and clean solution
            solution = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            solution = self._clean_solution(solution, prompt)
            solutions.append(solution)
            
        return solutions
    
    def _test_solutions(
        self,
        solutions: List[str],
        test_case: str
    ) -> List[Dict]:
        """Test multiple solutions against test case"""
        results = []
        
        for solution in solutions:
            result = {
                'syntax_valid': False,
                'passed': False,
                'execution_time': 0.0,
                'memory_usage': 0.0
            }
            
            # Check syntax validity
            try:
                compile(solution, '<string>', 'exec')
                result['syntax_valid'] = True
            except SyntaxError:
                results.append(result)
                continue
            
            # Execute test
            try:
                exec_result = self.execution_env.run_test(
                    solution=solution,
                    test_case=test_case
                )
                
                result.update({
                    'passed': exec_result['passed'],
                    'execution_time': exec_result['time'],
                    'memory_usage': exec_result['memory']
                })
                
            except Exception as e:
                print(f"Test execution error: {e}")
                
            results.append(result)
            
        return results
    
    def _clean_solution(self, solution: str, prompt: str) -> str:
        """Clean generated solution"""
        # Remove prompt from solution
        if solution.startswith(prompt):
            solution = solution[len(prompt):]
            
        # Extract function definition
        lines = solution.split('\n')
        cleaned_lines = []
        in_function = False
        
        for line in lines:
            if line.startswith('def '):
                in_function = True
            if in_function:
                cleaned_lines.append(line)
                
        return '\n'.join(cleaned_lines)
    
    def _compute_pass_k(self, passing_solutions: List[List[bool]], k: int) -> float:
        """Compute pass@k metric"""
        n = len(passing_solutions)
        c = 0
        
        for solutions in passing_solutions:
            if sum(solutions[:k]) > 0:  # At least one solution in first k passed
                c += 1
                
        return c / n

class ExecutionEnvironment:
    """Secure execution environment for testing code solutions"""
    def __init__(self, timeout: int = 5):
        self.timeout = timeout
        
    def run_test(self, solution: str, test_case: str) -> Dict:
        """Run test case in isolated environment"""
        import resource
        import time
        
        # Prepare complete test script
        test_script = f"""
{solution}

{test_case}
"""
        
        start_time = time.time()
        start_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        
        # Set resource limits
        resource.setrlimit(resource.RLIMIT_CPU, (self.timeout, self.timeout))
        resource.setrlimit(resource.RLIMIT_AS, (500 * 1024 * 1024, -1))  # 500MB memory limit
        
        try:
            # Execute test
            namespace = {}
            exec(test_script, namespace)
            passed = True
        except Exception as e:
            print(f"Test failed: {e}")
            passed = False
            
        # Measure resources
        end_time = time.time()
        end_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        
        return {
            'passed': passed,
            'time': end_time - start_time,
            'memory': (end_memory - start_memory) / 1024  # Convert to MB
        }