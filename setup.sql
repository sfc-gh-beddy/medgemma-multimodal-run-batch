-- ============================================================================
-- MedGemma ECG Batch Inference Demo - Setup Script
-- ============================================================================
-- Run this script BEFORE running the notebook
-- Requires ACCOUNTADMIN or equivalent privileges
-- ============================================================================

-- 1. Create Database and Schema
CREATE DATABASE IF NOT EXISTS MEDGEMMA_DEMO;
USE DATABASE MEDGEMMA_DEMO;
CREATE SCHEMA IF NOT EXISTS PUBLIC;
USE SCHEMA PUBLIC;

-- 2. Create Stages for ECG Images and Output
CREATE STAGE IF NOT EXISTS ECG_STAGE 
    DIRECTORY = (ENABLE = TRUE)
    COMMENT = 'Stage for ECG image files';

CREATE STAGE IF NOT EXISTS ECG_BATCH_OUTPUT_STAGE 
    DIRECTORY = (ENABLE = TRUE)
    COMMENT = 'Stage for batch inference output';

-- 3. Create Compute Pool for GPU Inference
-- NV_M provides 1x NVIDIA A10G GPU (24GB VRAM)
CREATE COMPUTE POOL IF NOT EXISTS MEDGEMMA_COMPUTE_POOL
    MIN_NODES = 1
    MAX_NODES = 1
    INSTANCE_FAMILY = GPU_NV_M
    AUTO_SUSPEND_SECS = 300
    COMMENT = 'GPU compute pool for MedGemma inference';

CREATE OR REPLACE NETWORK RULE ALLOW_ALL_NR
    MODE = EGRESS
    TYPE = HOST_PORT
    VALUE_LIST = ('0.0.0.0:443', '0.0.0.0:80');

CREATE OR REPLACE EXTERNAL ACCESS INTEGRATION ALLOW_ALL_EAI
    ALLOWED_NETWORK_RULES = (ALLOW_ALL_NR)
    ENABLED = TRUE
    COMMENT = 'External access for Kaggle, HuggingFace, and pip';

-- 4. Create Image Repository for Model Container
CREATE IMAGE REPOSITORY IF NOT EXISTS MEDGEMMA_DEMO.PUBLIC.IMAGE_REPO;

-- Verify Setup
SHOW STAGES IN SCHEMA MEDGEMMA_DEMO.PUBLIC;
SHOW COMPUTE POOLS LIKE 'MEDGEMMA_COMPUTE_POOL';
DESCRIBE COMPUTE POOL MEDGEMMA_COMPUTE_POOL;

SELECT 'Setup complete! Ready to run the notebook.' AS STATUS;
