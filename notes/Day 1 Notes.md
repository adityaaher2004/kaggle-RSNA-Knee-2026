# Day 1 Notes (Bio, Dataset)

#### Sources
1. CWRU, "MRI basics": `https://case.edu/med/neurology/NR/MRI%20Basics.htm`
Accessed 8/18/2026
2. GPT chats

## MRI, Bio Stuff

### MRI Working

- Magnetic field make random protons -> aligned protons
- RF disturbances / perturbations make them emit signals which measured
- FFT translate signal to brightness for image

### MRI planes
- Axial plane (transverse) : top-down / z-axis plane
- Sagittal plane : Side view / x or y axis
- Coronal plane (frontal) : front-facing (looking at you) / back-facing view (x or y axis)

### Some terminology
 - TR = Repetition time  
	 - Amount of time between successive pulse sequences applied to the same slice
- TE = Echo time
	- Amount of time between excitation and when we measure the MRI echo
- T1
	- Characteristic longitudinal recovery time
	- Time constant which determines the rate at which excited protons return to equilibrium
	- $M_Z(t) = M_0 \cdot (1 - e^{\frac{-t}{T_1}} )$
- T2
	- Time constant which determines the rate at which excited protons go out of phase with each other
	- Characteristic transverse dephasing time
	- $M_{x,y}(t) = M_{x,y}(0)\cdot e^{\frac{-t}{T_2}}$
- Relation to TR
	- If TR is short, longitudinal magnetization hasn't completely recovered before the next pulse
	- Short TR = strong Tl weighting
	- Long TR = weak TI weighting
- Relation to TE
	- If MRI measured early, TE≪T2​
	- If measured later, TE ~ T2
	- Long TE more T2 weighting
	- Short TE less T2 weighting
```
|               |                Short TE              |          Long TE            |
| ------------- | -------------------------------------|-----------------------------|
| Short TR      | T1 Weighted (fat bright, fluid dark) |           mixed             |
| Long TR       | Proton density weighted              |  T2 weighted (fluid bright) |
| ------------- | ------------------------------------ | ----------------------------|
```
## Interpreting the Dataset

1. The fundamental unit is a Study given by  `<StudyInstanceUID>`
2. Every study has N number of series inside it given by `<StudyInstanceUID>/SeriesA`, `<StudyInstanceUID>/SeriesB`, `<StudyInstanceUID>/SeriesC`
3. Every Series has K number of slices inside it
	- `SeriesA/slice1.dcm`, `SeriesA/slice2.dcm`, ... `SeriesA/sliceK.dcm`
4. Every Study has 12 binary labels for 12 knee problems.
_Some studies have labels explicitly provided in table, some labels need to be extracted from text reports_

`train.csv`
- Columns are: `<StudyInstanceUID>`, `Report`, `KneeProblemLabel_1`,...`KneeProblemLabel_12`

`train_series.csv` : bridge between studies and images
- There is **one row per MRI series**, not per study.
- Looks like:
```
| StudyInstaceUID | SeriesName | Fluid_Sensitive | Fat_suppresion | Anatomical_plane |
|------------------------------------------------------------------------------------|
| Study 1         | Series A   | 0               | 1              | Axial            |
| Study 1         | Series B   | 0               | 0              | Axial            |
| Study 1         | Series C   | 1               | 1              | Coronal          |
| Study 2         | Series A   | 1               | 1              | Axial            |
| Study 2         | Series B   | 0               | 0              | Sagittal         |
| Study 2         | Series C   | 1               | 0              | Coronal          |
| Study 2         | Series D   | 0               | 0              | Axial            |
```
### Test time
- Text report is available during training but not during testing
- Exploit the report **during training without making it necessary during inference**

#### ML Terminology
- Hierarchical attention
- Multi-instance learning
- Learning Using Privileged Information (LUPI)
- Multimodal training to unimodal inference
