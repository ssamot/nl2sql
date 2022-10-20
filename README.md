nl2sql
==============================

A sample NL2SQL solution

A sample `t5-small` finetuned on wikiSQL. The method used is fairly simple, we treat the problem as seq2Seq translation from natural language to an intermediate language. The project follows standard datascience cookiecutter naming patterns.

`make data_finetune`
`make train_finetune`
`make predict_finetune`

The output format is a small custom language I've to succinctly express SQL queries. A couple of examples from the validation dataset include: 

**Natural language question:** [START]-col0:Iowa State vs., col1:Overall Record, col2:in Ames, col3:at
Opponents Venue, col4:at Neutral Site, col5:Last 5 Meetings, col6:Last 10 Meetings, col7:Current
Streak, col8:Since Beginning of Big 12, ---When the value of "since beginning of big 12" is
synonymous with its' category, what are the in Ames values?-

**Actual SQL:** [START_SQL] 2:0:ci:[8]::oi:[0]:c['Since Beginning of Big 12']::
**Predicted SQL:** <pad> [START_SQL] 2:0:ci:[8]::oi:[0]:c["Sinda']::
---

**Natural language question:** [START]-col0:Year, col1:Division, col2:League, col3:Regular Season,
col4:Playoffs, col5:U.S. Open Cup, ---what's the u.s. open cup status for regular season of 4th,
atlantic division -

**Actual SQL:** [START_SQL] 5:0:ci:[3]::oi:[0]:c['4th, Atlantic Division']::
**Predicted SQL:** <pad> [START_SQL] 5:0:ci:[3]::oi:[0]:c['4th, Atlantic Division']::
