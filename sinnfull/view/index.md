# Visualization templates

The files in this folder are intended as templates. Their suggested usage is to copy them to another directory (e.g. “reports” or “labnotes”) and edit them to display results for the run(s) of interest.

:::{tip}  
Templates may use brackets (`{>`,`<}`) to indicate values that need to be set.  
:::

:::{tip}  
To generate a report from a single file, execute

    jb build <file name> --config <config file>

You can use [cell tags](https://jupyterbook.org/interactive/hiding.html) to hide code cells or remove them completely.
But if you use Bokeh plots, don't remove the cell with the Bokeh logo ! Otherwise your figures will not appear.
:::

[Result viewer](./Result%20viewer)
~ Summarize results from an ensemble of runs.
  - Run statistics (histograms of timestamps and run times)
  - Used parameter sets
  - Plot fit dynamics for the likelihood and parameters
  - Plot inferred latents
