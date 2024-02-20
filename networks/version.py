"""
Version record of models.
Author: Nan Lin
"""
__version__ = '0.4.0'

"""
Version 0.1.0:
    - date: 02-13-2024
    - feature: MaskEmbdMultiMPNV2 base

Version 0.2.0:
    - date: 02-13-2024
    - feature:
        - replaced the final_mp with an mlp `final_mlp` in MaskEmbdMultiMPNV2
        
Version 0.3.0:
    - date: 02-13-2024
    - feature:
        - use a group mlp for the final_mlp in MaskEmbdMultiMPNV2
        - reduced hidden_dim of edgeaggregatioin
        - use edgeaggregationv2 in MaskEmbdMultiMPNV2
        
Version 0.4.0:
    - date: 02-20-2024
    - feature:
        - add PFTAGConv as alternative to TAGConv
        - PFTAGConv create the half missing edges (TAGConv does not)
        - PFTAGConv can learn to use different edge weights for each k. 
        - use PFTAGConv in MaskEmbdMultiMPNV2
"""