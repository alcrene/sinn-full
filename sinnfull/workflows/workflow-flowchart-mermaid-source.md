flowchart TD
    subgraph modelels[model elements]
    Model
    Prior
    ObjectiveFunction
    end

    subgraph optimels[optim elements]
    Optimizer
    Recorder
    convtest[Convergence test]
    end

    Model -.- cmodel
    Prior -.- cprior
    ObjectiveFunction -.- cobj

    Optimizer -.- coptimizer
    Recorder -.- crec
    convtest -.- ctest

    subgraph workflow[Optimize Task Workflow]
    diskdata[on-disk data]
    synthdata[synthetic data]
    cdata{{create data accessor}}
    csampler{{create data\nsegment sampler}}
    cmodel{{create model}}
    cprior([create prior])
    cinit([choose initial params])
    cobj([choose objective])
    chyper([choose hyperparamters])
    coptimizer{{create optimizer}}
    crec[create recorders]
    ctest([create convergence tests])
    coptimize{{optimize model}}
    diskdata -.-> cdata
    synthdata -.-> cdata
    cprior -.-> synthdata
    cmodel -.-> synthdata
    cprior --> coptimizer
    chyper -.->vhyper([validate hyperparameters])
    cdata --> csampler
    csampler & cinit & cmodel & cobj & vhyper --> coptimizer
    coptimizer & crec & ctest --> coptimize
    end

    workflow --- blank[" "]
    blank -.->|execute in notebook| run[run optimize task]
    blank -.->|"execute as script"| cfile[create task file]
    cfile -.-> runlocal[run task on local machine]
    cfile -.->|"copy to server"| runserver[run task on server]

    style coptimize fill:#cde4ff, stroke:#147eff, stroke-width:2px
    style modelels fill:#f4f4f4, stroke:none
    style optimels fill:#f4f4f4, stroke:none
    style workflow font-weight:bold
    style blank fill:none, stroke:none
