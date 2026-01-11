# LongDA Dataset Card

## Dataset Description

**LongDA** is a data analysis benchmark for evaluating LLM-based agents under documentation-intensive analytical workflows. It features authentic U.S. government survey data with complete, long documentation, testing LLMs' ability to navigate complex real-world datasets before performing analysis.

### Dataset Summary

- **505 queries** extracted from 30 expert-written publications
- **17 U.S. national surveys** covering health, labor, economics, education, and social sciences
- **~263K tokens** average context per query (substantially longer than existing benchmarks)
- **Real-world analytical tasks** requiring multi-step reasoning and code execution

### Key Features

1. **Long Documentation**: Multiple unstructured documents per survey (codebooks, technical reports, user guides)
2. **Expert-Grounded Queries**: All queries extracted from real publications by domain experts
3. **Complex Data**: Large-scale tabular data with thousands of columns requiring careful navigation
4. **Authentic Workflow**: Mirrors real analytical practice where documentation navigation is the primary bottleneck

### Languages

- **Code**: Python
- **Documentation**: English
- **Data**: Numeric and categorical U.S. survey data

## Dataset Structure

```
benchmark/
├── benchmark.csv          # 505 queries with ground truth answers
└── [SURVEY]/             # 17 survey folders
    ├── data/             # Raw survey data files
    │   └── *.csv/*.dat   # Various formats
    └── docs/             # Long documentation
        └── *.pdf/*.txt   # Codebooks, user guides, technical reports
```

### Surveys Included

**Health**: NHANES (National Health and Nutrition Examination Survey), NHIS (National Health Interview Survey)

**Labor & Economics**: CPS-ASEC (Current Population Survey), ATUS (American Time Use Survey)

**Social Sciences**: GSS (General Social Survey), NSDUH (National Survey on Drug Use and Health), NSFG (National Survey of Family Growth)

**Science & Engineering**: NSCG (National Survey of College Graduates), HERD (Higher Education R&D Survey), SDR (Survey of Doctorate Recipients), SSERF (Survey of Science and Engineering Research Facilities)

**Government Operations**: ASFIN (Annual Survey of State Government Finances), ASPEP (Annual Survey of Public Employment & Payroll), ASPP (Annual Survey of Public Pensions), STC (State Tax Collections), RHFS (Residential Finance Survey), NTEWS (National Teacher and Principal Survey)

### Data Fields

**benchmark.csv** contains:
- `survey`: Survey acronym (e.g., NHANES, CPS-ASEC)
- `source`: Source publication title
- `internal_id`: Question number within the publication
- `query`: Natural language analytical query
- `answer_structure`: Expected format (`single_number` or list structure)
- `additional_info`: Context, units, and special requirements
- `answer`: Ground truth answer (verified against official publications)

## Dataset Creation

### Source Data

All data comes from publicly available U.S. government surveys spanning:
- **Health**: Population health, nutrition, healthcare access
- **Labor & Economics**: Employment, income, time use
- **Social Sciences**: Demographics, drug use, family structure
- **Science & Engineering**: Workforce, research funding, facilities
- **Government Operations**: State finances, employment, pensions

### Curation Process

1. **Survey Selection**: Chose 17 diverse, well-documented national surveys from 6 federal agencies
2. **Publication Collection**: Gathered 30 expert-written reports and publications
3. **Query Extraction**: Manually extracted 505 queries grounded in real analytical practice
4. **Ground Truth Verification**: Validated all answers against official statistics
5. **Documentation Assembly**: Included all relevant survey documentation (codebooks, guides, technical reports)

## Considerations for Using the Data

### Social Impact

This benchmark uses real government survey data covering sensitive topics including health, income, drug use, and demographics. Users should:
- Respect data privacy and usage policies
- Be aware of potential biases in survey data and sampling methods
- Use results responsibly when reporting findings
- Consider the ethical implications of automated data analysis

### Limitations

- Queries focus on U.S. data and may not generalize to other contexts
- Requires significant computational resources (long context windows ~263K tokens)
- Some surveys have complex sampling weights and methodologies
- Documentation navigation is challenging even for humans

## Additional Information

### Licensing Information

- **Code and Benchmark**: MIT License
- **Survey Data**: Public domain (U.S. government data)
- **Documentation**: Public domain (U.S. government publications)

### Citation

```bibtex
@article{li2025longda,
  title={LongDA: Benchmarking LLM Agents for Long-Document Data Analysis},
  author={Li, Yiyang and Zhang, Zheyuan and Ma, Tianyi and Wang, Zehong and Murugesan, Keerthiram and Zhang, Chuxu and Ye, Yanfang},
  journal={arXiv preprint arXiv:2501.XXXXX},
  year={2025}
}
```

### Contributions

Dataset curated by researchers at University of Notre Dame, IBM Research, and University of Connecticut. We thank all U.S. government agencies for making these valuable datasets publicly available.

For questions or issues, please visit: https://github.com/your-username/LongDA
