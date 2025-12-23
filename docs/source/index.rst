Vehicle-based Driver Drowsiness Detection
==========================================

This project provides analysis pipelines and models for
vehicle dynamics-based driver drowsiness detection (DDD).

It benchmarks lightweight ML models using vehicle-based features,
with emphasis on domain generalization across subjects.

**Key Features:**

- Multi-modal feature extraction (vehicle dynamics, EEG, physiological)
- Classical ML models with Optuna hyperparameter optimization
- Imbalanced learning strategies (SMOTE, ADASYN, etc.)
- Cross-subject domain generalization analysis
- HPC batch execution support

Getting Started
---------------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   guide/installation
   guide/quickstart
   guide/configuration

User Guides
-----------

.. toctree::
   :maxdepth: 2
   :caption: User Guides

   guide/data_pipeline
   guide/models
   guide/evaluation
   guide/imbalance_methods
   guide/experiments

Advanced Topics
---------------

.. toctree::
   :maxdepth: 2
   :caption: Advanced Topics

   guide/developer_guide
   guide/analysis
   guide/domain_generalization_pipeline
   guide/ranking_methods
   guide/evaluation_metrics

API Reference
-------------

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/modules

Indices and Tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
