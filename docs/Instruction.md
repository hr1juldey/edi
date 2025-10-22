# SCRIPT FOR AI DECOMPOSITION AGENT

# TARGET: Decompose PRD.md, HLD.md, LLD.md into a linked markdown structure

# EXECUTION CONTEXT: Assumes a 'decomposition-agent' CLI tool is available

# ---

# PHASE 1: INITIALIZATION

# ---

# Action: Create root directory and index.md

# Source Docs: PRD.md (for Executive Summary), HLD.md (for Architecture Overview and Component List)

decomposition-agent init \
  --root-dir ./edi_decomposed \
  --prd-file ./PRD.md \
  --hld-file ./HLD.md

# Expected output of 'init'

# 1. Creates directory './edi_decomposed'

# 2. Creates './edi_decomposed/index.md' containing

# - Product Name and Executive Summary from PRD

# - Architecture diagram from HLD

# - A "System Components" section with links to

# - [TUI Layer](./tui_layer.md)

# - [Vision Subsystem](./vision_subsystem.md)

# - [Reasoning Subsystem](./reasoning_subsystem.md)

# - [Orchestrator](./orchestrator.md)

# - [Storage Layer](./storage_layer.md)

# - [Integration Layer](./integration_layer.md)

# ---

# PHASE 2: TOP-LEVEL COMPONENT DECOMPOSITION

# ---

# Action: Create markdown files for each main component identified in HLD

# Source Docs: HLD.md (for component responsibilities), LLD.md (for child modules)

decomposition-agent create-component-doc --name "TUI Layer" --parent-doc ./edi_decomposed/index.md --hld-file ./HLD.md --lld-file ./LLD.md
decomposition-agent create-component-doc --name "Vision Subsystem" --parent-doc ./edi_decomposed/index.md --hld-file ./HLD.md --lld-file ./LLD.md
decomposition-agent create-component-doc --name "Reasoning Subsystem" --parent-doc ./edi_decomposed/index.md --hld-file ./HLD.md --lld-file ./LLD.md
decomposition-agent create-component-doc --name "Orchestrator" --parent-doc ./edi_decomposed/index.md --hld-file ./HLD.md --lld-file ./LLD.md --unknown-libs "DSpy"
decomposition-agent create-component-doc --name "Storage Layer" --parent-doc ./edi_decomposed/index.md --hld-file ./HLD.md --lld-file ./LLD.md
decomposition-agent create-component-doc --name "Integration Layer" --parent-doc ./edi_decomposed/index.md --hld-file ./HLD.md --lld-file ./LLD.md

# Expected output of 'create-component-doc' for "Vision Subsystem"

# 1. Creates './edi_decomposed/vision_subsystem.md'

# 2. Content includes

# - Upstream link to index.md

# - Purpose from HLD "Component Responsibilities" table

# - A "Sub-modules" section with links to

# - [sam_analyzer.py](./vision/sam_analyzer.md)

# - [clip_labeler.py](./vision/clip_labeler.md)

# - [scene_builder.py](./vision/scene_builder.md)

# - [change_detector.py](./vision/change_detector.md)

# - [models.py](./vision/models.md)

# ---

# PHASE 3: RECURSIVE DECOMPOSITION (MODULE & FUNCTION LEVEL)

# ---

# Action: Decompose a module file into its constituent functions

# This process is repeated recursively until all files in LLD.md are represented

# The agent should autonomously walk the file tree from LLD.md

# Example for a single module 'sam_analyzer.py'

decomposition-agent decompose-module \
  --source-file ./edi/vision/sam_analyzer.py \
  --parent-doc ./edi_decomposed/vision_subsystem.md \
  --output-dir ./edi_decomposed/vision/ \
  --lld-file ./LLD.md

# Expected output of 'decompose-module' for 'sam_analyzer.py'

# 1. Creates directory './edi_decomposed/vision/'

# 2. Creates './edi_decomposed/vision/sam_analyzer.md'

# 3. Content includes

# - Upstream link to vision_subsystem.md

# - Purpose from LLD comments for sam_analyzer.py

# - A "Functions" section with a link to

# - [analyze(image_path)](./sam_analyzer/analyze.md)

# ---

# PHASE 4: FUNCTION-LEVEL DOCUMENTATION

# ---

# Action: Create the final, most granular markdown file for a single function

# Source Docs: LLD.md (for signature), HLD.md (for detailed logic), PRD.md (for user story)

decomposition-agent create-function-doc \
  --function-name "analyze" \
  --class-name "SAMAnalyzer" \
  --module-path "edi/vision/sam_analyzer.py" \
  --parent-doc ./edi_decomposed/vision/sam_analyzer.md \
  --output-file ./edi_decomposed/vision/sam_analyzer/analyze.md \
  --prd-file ./PRD.md \
  --hld-file ./HLD.md \
  --lld-file ./LLD.md

# Expected output of 'create-function-doc'

# 1. Creates directory './edi_decomposed/vision/sam_analyzer/'

# 2. Creates './edi_decomposed/vision/sam_analyzer/analyze.md'

# 3. Content includes

# - Title: SAMAnalyzer.analyze()

# - Upstream link to sam_analyzer.md

# - Related User Story: "As a user, I want EDI to understand my image's composition..." (from PRD)

# - Function Signature: `analyze(image_path: str) -> List[Mask]` (from LLD/HLD)

# - Parameters: Detailed description of `image_path`

# - Returns: Detailed description of `List[Mask]`

# - Step-by-step Logic: Plain English description of the processing steps, including caching and OOM handling (from LLD/HLD)

# - Input/Output Data Structures: Markdown definition for the `Mask` object if it's a custom type

# ---

# AI ACTION INJECTION

# ---

# Action: When an unknown library is detected (e.g., in 'Orchestrator' or 'TUI Layer'), inject a request for information

# This is a flag on the create/decompose commands, as shown in Phase 2 for Orchestrator

# Example for 'intent_parser.py' which uses 'dspy'

decomposition-agent decompose-module \
  --source-file ./edi/reasoning/intent_parser.py \
  --parent-doc ./edi_decomposed/reasoning_subsystem.md \
  --output-dir ./edi_decomposed/reasoning/ \
  --lld-file ./LLD.md \
  --unknown-libs "DSpy"

# Expected output: 'intent_parser.md' will contain an additional markdown block

# ### AI Action Required: External Library Investigation

# This module uses the **DSpy** library

# 1.  **Request for Information:** Provide a link to the official documentation for `DSpy`

# 2.  **Search and Test:** I will search for examples of `dspy.Signature` and `dspy.ChainOfThought` to verify my understanding before implementation

# ---

# RECURSION COMPLETE

# ---

# The agent will continue this pattern, traversing the entire file structure outlined in LLD.md

# until every file, class, and function has a corresponding, interlinked markdown document
