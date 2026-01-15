"""Text generation utilities for benchmarking.

This module provides text generation tools optimized for multi-core processing,
generating diverse, meaningful Chinese text samples for benchmarking purposes.
"""

import random
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from typing import Optional
from tclogger import logger, logstr


# Chinese text templates - meaningful sentence patterns
CN_TEMPLATES = [
    "今天的{noun}真{adj}，让人感到{emotion}。",
    "在{place}里，我看到了{noun}，它{verb}得很{adv}。",
    '{person}说："{quote}"，这让我深受启发。',
    "关于{topic}的研究表明，{finding}是非常重要的。",
    "每当{time}，我就会想起{memory}，那是一段{adj}的回忆。",
    "这款{product}的特点是{feature}，适合{target}使用。",
    "根据{source}的报道，{news}引起了广泛关注。",
    "在{field}领域，{method}被认为是{evaluation}的方法。",
    "{action}的时候，需要注意{caution}，否则可能会{consequence}。",
    "我认为{opinion}，因为{reason}。",
    "虽然{challenge}很困难，但是{solution}可以解决这个问题。",
    "{season}的{place}非常{adj}，是{activity}的好时机。",
    "学习{skill}需要{quality}，更需要{effort}。",
    "这部{work}讲述了{plot}的故事，非常{evaluation}。",
    "如果你想{goal}，那么{advice}是个好建议。",
]

CN_NOUNS = [
    "风景",
    "天气",
    "音乐",
    "美食",
    "故事",
    "技术",
    "创意",
    "设计",
    "系统",
    "方案",
    "项目",
    "产品",
    "服务",
    "体验",
    "文化",
    "艺术",
    "科学",
    "教育",
    "健康",
    "环境",
    "电影",
    "书籍",
    "游戏",
    "软件",
    "网站",
    "应用",
    "算法",
    "模型",
    "数据",
    "信息",
]

CN_ADJS = [
    "美丽",
    "精彩",
    "独特",
    "重要",
    "有趣",
    "实用",
    "高效",
    "智能",
    "创新",
    "优秀",
    "温暖",
    "清新",
    "宁静",
    "活跃",
    "深刻",
    "简洁",
    "完善",
    "丰富",
    "稳定",
    "灵活",
]

CN_VERBS = [
    "运行",
    "工作",
    "发展",
    "进步",
    "创造",
    "设计",
    "实现",
    "优化",
    "分析",
    "处理",
    "学习",
    "探索",
    "研究",
    "发现",
    "建设",
    "管理",
    "服务",
    "支持",
    "提升",
    "改进",
]

CN_PLACES = [
    "城市",
    "公园",
    "图书馆",
    "博物馆",
    "校园",
    "办公室",
    "实验室",
    "工厂",
    "商场",
    "医院",
    "山区",
    "海边",
    "乡村",
    "社区",
    "广场",
    "车站",
    "机场",
    "港口",
    "街道",
    "市场",
]

CN_PERSONS = [
    "老师",
    "专家",
    "学者",
    "工程师",
    "设计师",
    "医生",
    "作家",
    "艺术家",
    "科学家",
    "企业家",
    "朋友",
    "同事",
    "领导",
    "前辈",
    "教授",
    "研究员",
    "开发者",
    "分析师",
    "顾问",
    "记者",
]

CN_TOPICS = [
    "人工智能",
    "机器学习",
    "自然语言处理",
    "计算机视觉",
    "数据科学",
    "云计算",
    "物联网",
    "区块链",
    "网络安全",
    "软件工程",
    "用户体验",
    "产品设计",
    "市场营销",
    "项目管理",
    "健康管理",
    "环境保护",
    "教育创新",
    "文化传承",
    "科技发展",
    "社会进步",
]

CN_EMOTIONS = [
    "高兴",
    "感动",
    "欣慰",
    "惊喜",
    "满足",
    "期待",
    "兴奋",
    "平静",
    "温馨",
    "舒适",
]

CN_ADVS = [
    "认真",
    "努力",
    "仔细",
    "积极",
    "耐心",
    "专注",
    "用心",
    "勤奋",
    "热情",
    "细致",
]

CN_TIMES = [
    "清晨",
    "傍晚",
    "深夜",
    "周末",
    "假期",
    "春天",
    "夏天",
    "秋天",
    "冬天",
    "节日",
]

CN_SEASONS = ["春天", "夏天", "秋天", "冬天"]

CN_ACTIVITIES = [
    "旅行",
    "运动",
    "摄影",
    "阅读",
    "写作",
    "绘画",
    "音乐",
    "舞蹈",
    "烹饪",
    "园艺",
]

CN_QUOTES = [
    "创新是进步的源泉",
    "坚持不懈才能成功",
    "知识改变命运",
    "实践出真知",
    "细节决定成败",
    "团结就是力量",
    "学无止境",
    "厚积薄发",
]

CN_TARGETS = ["专业人士", "学生", "企业", "个人", "团队", "初学者", "研究人员"]
CN_SOURCES = ["最新研究", "行业报告", "专家分析", "数据调查", "市场研究"]
CN_METHODS = ["数据驱动", "系统化", "模块化", "迭代式", "敏捷", "智能化"]
CN_CAUTIONS = ["安全性", "可靠性", "兼容性", "效率", "质量", "成本"]
CN_CONSEQUENCES = ["影响效果", "造成问题", "带来风险", "产生误差", "降低效率"]
CN_EFFORTS = ["持续的练习", "深入的思考", "不断的尝试", "系统的学习"]
CN_WORKS = ["作品", "电影", "书籍", "文章", "报告", "研究"]
CN_GOALS = ["提升能力", "实现目标", "获得成功", "解决问题", "创造价值"]
CN_ADVICES = [
    "保持专注和耐心",
    "制定明确的计划",
    "不断学习和实践",
    "寻求专业指导",
    "善于总结经验",
]


def _fill_cn_template(rng: random.Random, template: str) -> str:
    """Fill a Chinese template with random words.

    Args:
        rng: Random number generator instance
        template: Template string with placeholders

    Returns:
        Filled template string
    """
    result = template

    # Helper functions for complex replacements
    def gen_finding():
        return f"{rng.choice(CN_ADJS)}的{rng.choice(CN_NOUNS)}"

    def gen_memory():
        return f"在{rng.choice(CN_PLACES)}{rng.choice(CN_VERBS)}的日子"

    def gen_feature():
        return f"{rng.choice(CN_ADJS)}且{rng.choice(CN_ADJS)}"

    def gen_news():
        return f"{rng.choice(CN_TOPICS)}的{rng.choice(CN_ADJS)}发展"

    def gen_method():
        return f"{rng.choice(CN_METHODS)}方法"

    def gen_opinion():
        return f"{rng.choice(CN_TOPICS)}非常{rng.choice(CN_ADJS)}"

    def gen_reason():
        return f"它能够{rng.choice(CN_VERBS)}并带来{rng.choice(CN_ADJS)}的效果"

    def gen_challenge():
        return f"{rng.choice(CN_TOPICS)}中的问题"

    def gen_solution():
        return f"采用{rng.choice(CN_ADJS)}的{rng.choice(CN_NOUNS)}"

    def gen_plot():
        return (
            f"{rng.choice(CN_PERSONS)}在{rng.choice(CN_PLACES)}{rng.choice(CN_VERBS)}"
        )

    replacements = {
        "{noun}": lambda: rng.choice(CN_NOUNS),
        "{adj}": lambda: rng.choice(CN_ADJS),
        "{verb}": lambda: rng.choice(CN_VERBS),
        "{place}": lambda: rng.choice(CN_PLACES),
        "{person}": lambda: rng.choice(CN_PERSONS),
        "{topic}": lambda: rng.choice(CN_TOPICS),
        "{emotion}": lambda: rng.choice(CN_EMOTIONS),
        "{adv}": lambda: rng.choice(CN_ADVS),
        "{time}": lambda: rng.choice(CN_TIMES),
        "{season}": lambda: rng.choice(CN_SEASONS),
        "{activity}": lambda: rng.choice(CN_ACTIVITIES),
        "{quote}": lambda: rng.choice(CN_QUOTES),
        "{finding}": gen_finding,
        "{memory}": gen_memory,
        "{product}": lambda: rng.choice(CN_NOUNS),
        "{feature}": gen_feature,
        "{target}": lambda: rng.choice(CN_TARGETS),
        "{source}": lambda: rng.choice(CN_SOURCES),
        "{news}": gen_news,
        "{field}": lambda: rng.choice(CN_TOPICS),
        "{method}": gen_method,
        "{evaluation}": lambda: rng.choice(CN_ADJS),
        "{action}": lambda: rng.choice(CN_VERBS),
        "{caution}": lambda: rng.choice(CN_CAUTIONS),
        "{consequence}": lambda: rng.choice(CN_CONSEQUENCES),
        "{opinion}": gen_opinion,
        "{reason}": gen_reason,
        "{challenge}": gen_challenge,
        "{solution}": gen_solution,
        "{skill}": lambda: rng.choice(CN_TOPICS),
        "{quality}": lambda: rng.choice(CN_ADJS),
        "{effort}": lambda: rng.choice(CN_EFFORTS),
        "{work}": lambda: rng.choice(CN_WORKS),
        "{plot}": gen_plot,
        "{goal}": lambda: rng.choice(CN_GOALS),
        "{advice}": lambda: rng.choice(CN_ADVICES),
    }

    for key, value_fn in replacements.items():
        if key in result:
            result = result.replace(key, value_fn(), 1)

    return result


def _generate_batch(args: tuple) -> list[str]:
    """Generate a batch of text samples (worker function for multiprocessing).

    Args:
        args: Tuple of (batch_count, min_len, max_len, seed, start_counter)

    Returns:
        List of generated text samples
    """
    batch_count, min_len, max_len, seed, start_counter = args

    # Each worker has its own RNG with unique seed
    rng = random.Random(seed)
    samples = []
    counter = start_counter

    for _ in range(batch_count):
        # Generate text from template
        template = rng.choice(CN_TEMPLATES)
        text = _fill_cn_template(rng, template)

        # Add unique suffix using counter to ensure uniqueness
        counter += 1
        suffix = f" [{counter}]"
        text = text + suffix

        # Adjust length if needed
        current_len = len(text)
        if current_len < min_len:
            # Extend by adding complete new templates
            while len(text) < min_len:
                new_template = rng.choice(CN_TEMPLATES)
                extra = " " + _fill_cn_template(rng, new_template)
                text += extra
        elif current_len > max_len:
            # Truncate
            text = text[: max_len - 3] + "..."

        samples.append(text)

    return samples


class TEIBenchTextGenerator:
    """Generate meaningful, non-repeating Chinese texts.

    Uses template-based generation with random word substitution to create
    diverse, meaningful text samples. Optimized for multi-core processing.
    """

    def __init__(self, seed: int = 42, num_workers: Optional[int] = None):
        """Initialize the text generator.

        Args:
            seed: Random seed for reproducibility
            num_workers: Number of worker processes (default: CPU count, max 8)
        """
        self.seed = seed
        self.rng = random.Random(seed)

        # Determine optimal number of workers
        cpu_count = mp.cpu_count()
        if num_workers is None:
            # Default to min(cpu_count, 8) for optimal performance
            self.num_workers = min(cpu_count, 8)
        else:
            self.num_workers = min(num_workers, cpu_count)

        self._counter = 0

    def _generate_one(self, min_len: int, max_len: int) -> str:
        """Generate a single text sample (for small counts or fallback).

        Args:
            min_len: Minimum character length
            max_len: Maximum character length

        Returns:
            A unique text sample
        """
        # Generate text from template
        template = self.rng.choice(CN_TEMPLATES)
        text = _fill_cn_template(self.rng, template)

        # Add unique suffix
        self._counter += 1
        suffix = f" [{self._counter}]"
        text = text + suffix

        # Adjust length
        current_len = len(text)
        if current_len < min_len:
            while len(text) < min_len:
                new_template = self.rng.choice(CN_TEMPLATES)
                extra = " " + _fill_cn_template(self.rng, new_template)
                text += extra
        elif current_len > max_len:
            text = text[: max_len - 3] + "..."

        return text

    def generate(
        self,
        count: int,
        min_len: int = 100,
        max_len: int = 300,
        show_progress: bool = True,
    ) -> list[str]:
        """Generate multiple unique text samples using multiprocessing.

        Args:
            count: Number of samples to generate
            min_len: Minimum character length per sample
            max_len: Maximum character length per sample
            show_progress: Whether to show progress bar

        Returns:
            List of unique text samples
        """
        logger.note(
            f"> Generating {count:,} text samples ({min_len}-{max_len} chars)..."
        )

        # For small counts, use single-threaded generation
        # Multiprocessing overhead not worth it for < 10000 samples
        if count < 10000 or self.num_workers <= 1:
            return self._generate_sequential(count, min_len, max_len, show_progress)

        return self._generate_parallel(count, min_len, max_len, show_progress)

    def _generate_sequential(
        self,
        count: int,
        min_len: int,
        max_len: int,
        show_progress: bool,
    ) -> list[str]:
        """Generate samples sequentially (single-threaded)."""
        samples = []
        progress_interval = max(1, count // 10)

        for i in range(count):
            sample = self._generate_one(min_len, max_len)
            samples.append(sample)

            if show_progress and (i + 1) % progress_interval == 0:
                pct = (i + 1) / count * 100
                logger.mesg(f"  Progress: {pct:.0f}% ({i + 1:,}/{count:,})")

        self._log_stats(samples)
        return samples

    def _generate_parallel(
        self,
        count: int,
        min_len: int,
        max_len: int,
        show_progress: bool,
    ) -> list[str]:
        """Generate samples in parallel using multiprocessing."""
        import time

        logger.mesg(f"  Using {self.num_workers} worker processes...")
        start_time = time.perf_counter()

        # Split work evenly among workers
        batch_size = count // self.num_workers
        remainder = count % self.num_workers

        # Prepare arguments for each worker
        # Each worker gets: (batch_count, min_len, max_len, unique_seed, start_counter)
        worker_args = []
        current_counter = self._counter

        for i in range(self.num_workers):
            # Add remainder to first few workers
            worker_count = batch_size + (1 if i < remainder else 0)
            if worker_count == 0:
                continue

            # Each worker gets unique seed derived from base seed
            worker_seed = self.seed + i * 10007  # Prime number for better distribution

            worker_args.append(
                (
                    worker_count,
                    min_len,
                    max_len,
                    worker_seed,
                    current_counter,
                )
            )
            current_counter += worker_count

        # Update counter for future calls
        self._counter = current_counter

        # Execute in parallel
        samples = []
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all tasks
            futures = list(executor.map(_generate_batch, worker_args))

            # Collect results
            for batch_samples in futures:
                samples.extend(batch_samples)

        elapsed = time.perf_counter() - start_time
        logger.mesg(f"  Parallel generation completed in {elapsed:.2f}s")

        self._log_stats(samples)
        return samples

    def _log_stats(self, samples: list[str]) -> None:
        """Log statistics about generated samples."""
        total_chars = sum(len(s) for s in samples)
        avg_len = total_chars / len(samples) if samples else 0

        logger.okay(f"  Generated {len(samples):,} samples")
        logger.mesg(f"  Total chars: {total_chars:,}, avg length: {avg_len:.1f}")
