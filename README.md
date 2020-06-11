# SCAN: A Spatial Context Attentive Network for Joint Multi-Agent Intent Prediction
Code for NeurIPS20 Submission: SCAN: A Spatial Context Attentive Network for Joint Multi-Agent Intent Prediction

Predicting pedestrian intent is a complex problem because pedestrians are influenced by the current trajectories of their neighbors as well as their expected future trajectories. Human navigation behavior is governed by implicit spatial rules such as yielding right-of-way, respecting personal space, etc. that typical sequence to sequence learning approaches cannot capture. In this work, we propose SCAN, a Spatial Context Attentive Network that can jointly predict socially-acceptable future trajectories of all pedestrians in a scene. 

Our proposed method uses a novel spatial attention mechanism to model the influence of spatially close neighbors on each other and a temporal attention mechanism to model the ability of pedestrians to respond similarly to previously experienced spatial contexts or situations. 

## Our Model

Our model contains an LSTM-based Encoder-Decoder Framework that takes as input observed trajectories for all pedestrians in the frame and <em> jointly predicts </em> future trajectories for all the pedestrians in the given frame. To account for spatial influences of spatially close pedestrians on each other, our model uses a <em> Spatial Attention Mechanism </em> that infers and incorporates perceived spatial contexts into each pedestrian's LSTM's knowledge. In the decoder, our model additionally uses a temporal attention mechanism to attend to the observed spatial contexts for each pedestrian, to enable the model to learn how to navigate by learning from previously encountered spatial situations. 

<img src = https://github.com/pedestrian-intent/SCAN/blob/master/img/model.png
width="1200" height="500">

## Example Predictions
Below we show some socially acceptable trajectories predicted by our model in crowded environments. Every pedestrian navigating in the frame is denoted by a different color. The images on the left are the ground truth trajectories and the images on the left denote the trajectories predicted by our model. Our model is able to predict socially acceptable trajectories that also demonstrate social behavior such as walking in pairs, walking in groups, avoiding collision, and so on. 

<p align="center">
<img src = https://github.com/pedestrian-intent/SCAN/blob/master/img/Pass_collision_avoidance/ground_truth.gif
width="300" height="300"> 
<img src =https://github.com/pedestrian-intent/SCAN/blob/master/img/Pass_collision_avoidance/prediction.gif
width="300" height="300">
</p>

<p align="center">
<img src = https://github.com/pedestrian-intent/SCAN/blob/master/img/Pass_collision_avoidance_/ground_truth.gif
width="300" height="300"> 
<img src =https://github.com/pedestrian-intent/SCAN/blob/master/img/Pass_collision_avoidance_/prediction.gif
width="300" height="300">
</p>

<p align="center">
<img src = https://github.com/pedestrian-intent/SCAN/blob/master/img/Pass_collision_avoidance_2/ground_truth.gif
width="300" height="300"> 
<img src =https://github.com/pedestrian-intent/SCAN/blob/master/img/Pass_collision_avoidance_2/prediction.gif
width="300" height="300">
</p>

In some cases, our model fails to predict accurate trajectories. The primary reason for that is that our model is agnostic to scene context and hence does not contain information about useful scene features such as turns, building entrances, sidewalks, and so on. Below we show a fail case for our model that is experienced because of lack of scene information:

<p align="center">
<img src = https://github.com/pedestrian-intent/SCAN/blob/master/img/Fail_scene_context/ground_truth.gif
width="300" height="300"> 
<img src =https://github.com/pedestrian-intent/SCAN/blob/master/img/Fail_scene_context/prediction.gif
width="300" height="300">
</p>

## Training Details

We train and evaluate our model on five publicly available datasets: ETH, HOTEL, UNIV, ZARA1, ZARA2. We follow a leave-one-out process where we train on four of the five models and test on the fifth. The exact training, validation, test datasets we use are in directory <bash> data/ </bash>. For each pedestrian in a given frame, our model observes the trajectory for 8 time steps (3.2 seconds) and predicts intent over future 8 time steps (3.2 seconds) jointly for all pedestrians in the scene.  

To download the data:

```bash
sh scripts/download_data.sh
```

To train our models with our best-performing parameters to reproduce our results, run:

```bash
sh scripts/train.sh <model_name> <dataset_name> 16 32 128 16
```

where: `<model_name>` can be `spatial_temporal_model` or `spatial_model`. `spatial_temporal_model` is our final model, <em> SCAN </em>, and `<spatial_model>` is our ablation, <em> vanillaSCAN </em>. `<dataset_name>` choices are `zara1`, `zara2`, `univ`, `eth`, `hotel`. 

To evaluate <em> SCAN </em> on the five datasets, run:

```bash
sh scripts/test.sh <dataset_name> 
```
To evaluate <em> vanillaSCAN </em> change `spatial_temporal_model` to `spatial_model` in `scripts/test.sh`. 

We evaluate our models using 2 metrics: ADE (Average Displacement Error) and FDE (Final Displacement Error). 

|             	| ETH         	| HOTEL      	| ZARA1       	| ZARA2       	| UNIV        	| AVG         	|
|-------------	|-------------	|------------	|-------------	|-------------	|-------------	|-------------	|
| vanillaSCAN 	| 0.75 / 1.44 	| 0.45/ 0.91 	| 0.39 / 0.89 	| 0.34 / 0.75 	| 0.62 / 1.31 	| 0.51 / 1.06 	|
| SCAN        	| 0.55/0.78   	| 0.47 /0.90 	| 0.22/0.45   	| 0.31/0.67   	| 0.62 /1.28  	| 0.43/0.82   	|

## Qualitative Analysis

### Track Fragmentation Analysis

Noisy or missing data is a common problem in object tracking. It is important for an intent prediction model to be able to predict fairly accurate future trajectories despite interrupted observed ground truth trajectories. We evaluated the robustness of <em> SCAN </em> to such track fragmentation by randomly dropping frames from the observation window for each pedestrian. 

To evaluate track fragmentation add 

```bash
--drop_frames=<drop_frames> 
```

to `scripts/test.sh`. Where `<drop_frames>` is the number of frames or timesteps from the observed sequence that are to be dropped per pedestrian in the scene. 

### Collision Analysis

To demonstrate the capability of our spatial attention mechanism to predict safe, <em> socially-acceptable </em> trajectories, we evaluate the ability of trajectories predicted by our model to avoid <em> collisions </em>. For a given prediction time window of 12 timesteps and a certain distance threshold, we say the situation contains a collision if the distance between any two pedestrians in the frame drops below the distance threshold at any of the 12 timesteps. To evaluate <em> SCAN </em> and <em> vanillaSCAN </em> for socially acceptable trajectories and ability to avoid collisions, run:

```bash
sh scripts/collisions.sh <dataset_name> 
```


