---
authors: []
tags: []
title: source
---

# **Quarterly Uptime Report** 

## **Summary** 

This report summarizes service availability for the first quarter. All tracked services met their availability targets. Two services recorded minor incidents, neither of which breached the service level objective. 

## **Service Metrics** 

The table below lists uptime and incident counts for each tracked service over the quarter. Uptime is measured against the published maintenance windows. 

| Service| Uptime| Incidents|Notes|
|---|---|---|---|
|Gateway|99.95%|1|Brief restart during a deploy.|
|Database|99.99%|0|No incidents this quarter.|
|Cache|99.80%|2|Two evictions under peak load.|
|Search|99.90%|1|One reindex pause overnight.|

[table-rows]
[**Service Metrics**] Service=Gateway, Uptime=99.95%, Incidents=1, Notes=Brief restart during a deploy.
[**Service Metrics**] Service=Database, Uptime=99.99%, Incidents=0, Notes=No incidents this quarter.
[**Service Metrics**] Service=Cache, Uptime=99.80%, Incidents=2, Notes=Two evictions under peak load.
[**Service Metrics**] Service=Search, Uptime=99.90%, Incidents=1, Notes=One reindex pause overnight.
[/table-rows]


## **Notes** 

Incident counts include only events that paged the on-call engineer. Planned maintenance is excluded from the uptime figures.