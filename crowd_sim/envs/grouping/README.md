# Code overview
The file **grouping_functions.py** contains two classes:
* Grouper: clutsers and colors pedestrians for visualization. 
* GroupSpaceGenerator: contains methods for computing the perimeter points for visualizing the occupying space of clustered groups.

Within **crowd_sim/envs/crowd_sim.py**, the Grouper object is called during the 'update' function. The group space is visualized on a separate plot from the main environment, and separate 'update' functions are used for each space generation method, which can be specified at the bottom of the file. 

## Grouper
The grouper class stores the states, humans, groups, and group colors for the complete simulation iteration. Within this class, there are two methods for clustering pedestrians. 

**DBScan:** inspired by the work by [Wang et. al.](https://chrismavrogiannis.com/pdfs/wang2021groupbased.pdf), we employ DBScan (*sklearn* toolkit) to group pedestrians based on their position and velocity.

**Coherent Filter:** we adapt the work by [Zhou et al.](https://scispace.com/pdf/coherent-filtering-detecting-coherent-motions-from-crowd-xi8nhnf2s2.pdf), which considers the spatiotemporal relationship of pedestrians across frames to detect coherent motions. The implementation is based principles of *Coherent Neighbor Invariance*, that the neighborship of coherent motions tend to remain invariant over time, and the velocity correlations of neighboring individuals remain high over time.

## GroupSpaceGenerator
Within this class, there are two methods for computing the group space. Note: these implementations have been developed using the **Coherent Filter** grouping approach, and currently see less consistent results with DBScan (although this is likely a matter of tuning parameters).

**Convex Hull (social space):** We first compute personal space for each pedestrian based on the work of [Rachel Kirby](https://www.ri.cmu.edu/publications/social-robot-navigation/), then use the *scipy* toolbox to generate a convex hull containing all members of the group (including personal space).

**Perimeter Polygon:** We find the outermost members of each group and use their poses to generate a polygon encapsulating the remaining group members. Simple circles are used to generate personal space.

## Setup
Used Python 3.8.5 for development.

## Getting started
To select the methods for grouping and space generation, see inline comments in the code. 

Within **crowd_sim/envs/crowd_sim.py**, there are two major modifications to the original *ComplexityNav* **crowd_sim.py**. Use 
```
self.group_labels = self.dbscan_group(frame) 
```
for dbscan clustering, and 
```
self.group_labels = self.CoherentFilter(frame, 5, 3, 0.9)
```
for coherent filter. Use 

```
self.group_labels = self.dbscan_group(frame) 
```
update_chull(0)
anim2 = animation.FuncAnimation(fig2, update_chull, frames=len(self.states), interval=self.time_step * 1000)
'''
for the convex hull space generator, and 
update_complete_polygon(0)
anim2 = animation.FuncAnimation(fig2, update_complete_polygon, frames=len(self.states), interval=self.time_step * 1000)
'''
for the polynomial method.
