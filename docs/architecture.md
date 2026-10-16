`mermaid
graph TD
    indexer --> config
    indexer --> vector_store
    main --> config
    main --> indexer
    main --> query_pipeline
    query_pipeline --> config
    query_pipeline --> vector_store
    vector_store --> config
`
