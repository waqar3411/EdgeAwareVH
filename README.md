# EdgeAwareVH
## EdgeAware Graphical Neural Network for road-side object geo-location
### Dataset:
To train EdgeAwareVH, a synthetic dataset was generated over a 
$20\,\text{km} \times 20\,\text{km}$ region centered around Amsterdam at $52.381261$ lat, $4.867661$ lon.

### Abstract
Accurate geolocation of roadside objects from street-level imagery remains a challenging problem due to noisy detections, which are typical for real image datasets with limited annotations available. This leads to inaccurate camera-to-object distance and bearing estimates. Traditional triangulation methods struggle in these conditions as they rely heavily on clean geometric measurements and predefined assumptions about view consistency. To address this we propose a scalable Graph Neural Network (GNN) model, coined EdgeAwareVH, that learns to reason directly over view-hypothesis relationships. The model integrates edge-aware attention with geometric message passing, enabling it to selectively emphasize reliable observations and suppress spurious intersections. Our pipeline combines geometric cues such as depths, bearing angles, camera pose information and OpenStreetMap-derived spatial context providing robustness in complex urban environments. We introduce a realistic noise simulation pipeline that mimics the behavior of weakly supervised detectors and evaluate our method under multiple noise configurations. Our experiments with wastebins geolocation on noisy simulated and real-world data demonstrate that EdgeAwareVH achieves high localization accuracy and is resilient to depth and bearing noise.

## Proposed Method:
![Proposed_block](https://github.com/user-attachments/assets/49a9ffb6-5beb-40a7-ab0c-eba64ea88bce)


## Result Analysis:
<img width="415" height="228" alt="image" src="https://github.com/user-attachments/assets/9946d446-3ede-4728-8179-3c9d9f497948" />

<img width="415" height="196" alt="image" src="https://github.com/user-attachments/assets/fb57cbea-8a9b-4f29-a92a-1528fa22edfd" />

<img width="398" height="141" alt="image" src="https://github.com/user-attachments/assets/a47522c7-f24f-4d56-849f-ecdd99fb2705" />

## Conclusion:
We introduce EdgeAwareVH, a scalable and geometry-driven GNN approach for robust roadside object geolocation from weekly supervised street-level imagery. Practically, any existing GIS record of the asset of interest can be used to train the model without the need for costly image matching annotations. By formulating the triangulation as view-hypothesis graph and integrating the edge-aware attention with geometric message passing, the proposed method effectively distinguishes viable from spurious intersections. The incorporation of the OSM-derived information further enhances the model performance by penalizing hypotheses inconsistent with the spatial context. Our experiments with wastebins geolocation show that EdgeAwareVH outperforms both traditional geometric baseline as well as attention-based alternatives. The future work may focus on incorporating self-attention blocks to strengthen view-to-view reasoning and further enhance robustness.

