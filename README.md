# DEService (Discrete (but also Discreet) Event Simulator Service)

## Goal
- Deploy the simulator as a scalable service on the cloud

#### ToDo
- Clean up modules in DEService/lib, currently everythings in one module, need to resolve importing into each other. Genie recursively loads all in this folder, figure out how to make it not complain when loading in separate modules with dependencies between them.
- Standardise what endpoints expect to recieve and return