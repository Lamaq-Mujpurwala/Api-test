import pandas as pd
from conceptexpert import new_concept_agent_executor
from datasetexpert import dataset_expert_chain
from domainexpert import dsgroq_chain
from langchain_groq import ChatGroq

def generate_case_study_v2(csv_path, concept):
    # Data ingestion and preparation
    df = pd.read_csv(csv_path,on_bad_lines='skip',engine='python')
    data_sample = df.head(20).to_string()
    corr_matrix = df.corr(numeric_only=True).to_string()

    # Agent analyses
    dataset_analysis = dataset_expert_chain.invoke({"concept": concept, "dataset": data_sample})
    domain_analysis = dsgroq_chain.invoke({"dataset_summary": data_sample, "correlation_stats": corr_matrix})
    concept_resources = new_concept_agent_executor.invoke({"concept": concept})['output']

    # Final structured guide generation
    final_llm = ChatGroq(
        api_key="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",
        model="llama-3.3-70b-versatile",
        temperature=0.5
    )

    structured_prompt = f"""
    Create a step-by-step learning guide using this format (ALWAYS USE THIS FORMAT STRICTLY):

    # Case Study: Learning {concept} with {csv_path.split('/')[-1]}
    
    ## Foundational Understanding
    Use foundational knowledge from {concept_resources} to explain the concept.

    ## Practical Implementation Guide
    
    Generate 3-5 sequential learning steps following this structure for each:
    Step [N]. [Step Title] 
    - **Concept**: 1-sentence explanation
    - **Analogy**: Real-world comparison (if complex concept)
    - **Dataset Connection**: How this relates to {csv_path.split('/')[-1]} columns: {df.columns.tolist()}
    - **Task**: Practical exercise using the dataset
    - **Resources**: 
      Use relevent resources for each step from {concept_resources}
      Point to relevant video from {concept_resources} (Do not include video link in the step)

    Include these insights from analysis:
    - Domain Expert Findings: {domain_analysis}
    - Dataset Characteristics: {dataset_analysis}

    Required elements:
    - Numbered steps with clear progression
    - Actual YouTube links from resources
    - Dataset-specific examples in tasks
    - Common pitfalls warning section
    - Final implementation checklist
    """

    return final_llm.invoke(structured_prompt).content
