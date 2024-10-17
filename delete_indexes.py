from elasticsearch import Elasticsearch

# Create an Elasticsearch client with the options method
es = Elasticsearch(request_timeout=1000, hosts=["http://localhost:9200"]).options(
    ignore_status=[400, 404]
)

# Delete all indexes
es.indices.delete(index="_all")
