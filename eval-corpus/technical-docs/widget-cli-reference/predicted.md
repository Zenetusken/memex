---
authors: []
tags: []
title: source
---

# **Widget CLI Reference** 

## **Installation** 

Install the Widget CLI from the package index. The CLI requires a recent runtime and writes its configuration to the user config directory. 

```
pip install widget-cli
widget --version
```

## **Commands** 

The CLI groups its functionality into subcommands. Each subcommand accepts its own flags; global flags are accepted before the subcommand name. 

### **widget build** 

Compile the project into a deployable artifact. The build step reads the manifest in the working directory and writes output to the build folder. 

```
widget build --target prod --out ./dist
```

#### **Build options** 

The build subcommand accepts a small set of options: 

- --target selects the build profile (dev or prod). 

- --out overrides the output directory. 

- --clean removes prior artifacts before building. 

### **widget deploy** 

Push a built artifact to a named environment. Deployment is atomic and rolls back automatically if the health check fails. 

```
widget deploy --env staging --artifact ./dist/app.tar
```