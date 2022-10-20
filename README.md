nl2sql
==============================

A sample `t5-small` finetuned on wikiSQL. The method used is fairly simple, we treat the problem as seq2Seq translation from natural language to an intermediate language, in the spirit of SeeD (https://arxiv.org/abs/2105.07911). The project follows standard datascience cookiecutter naming patterns. Make targets are as follows.

`make data_finetune`
`make train_finetune`
`make predict_finetune`

The output format is a small custom language I've created to succinctly express SQL queries. A couple of examples from the validation dataset are:


**Natural language question:** `[START]-col0:District, col1:Total amount of trees, col2:Prevailing types,
%, col3:Amount of old trees, col4:Amount of trees, that require replacement, ---What is the district
when prevailing types, % is acer negundo — 30.22 tilia — 18.6 poplar — 15.23?-`

**Actual SQL:** `[START_SQL] 0:0:ci:[2]::oi:[0]:c['Acer negundo — 30.22 Tilia — 18.6 Poplar — 15.23']::`

**Predicted SQL:** `<pad> [START_SQL] 0:0:ci:[2]::oi:[0]:c['Acer Negundo — 30.22 Tilia — 18.6 Poplar — 15.23']::`


---

**Natural language question:** `[START]-col0:Year, col1:Division, col2:League, col3:Regular Season,
col4:Playoffs, col5:U.S. Open Cup, ---what's the u.s. open cup status for regular season of 4th,
atlantic division -`

**Actual SQL:** `[START_SQL] 5:0:ci:[3]::oi:[0]:c['4th, Atlantic Division']::`

**Predicted SQL:** `<pad> [START_SQL] 5:0:ci:[3]::oi:[0]:c['4th, Atlantic Division']::`
`
