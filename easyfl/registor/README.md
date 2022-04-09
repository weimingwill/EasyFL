# Docker Register

docker-register sets up a container running [docker-gen][1].  docker-gen dynamically generate a
python script when containers are started and stopped.  This generated script registers the running
containers host IP and port in etcd with a TTL.  It works in tandem with docker-discover which
generates haproxy routes on the host to forward requests to registered containers.

Together, they implement [service discovery][2] for docker containers with a similar architecture
to [SmartStack][3]. docker-register is analagous to [nerve][4] in the SmartStack system.

See also [Docker Service Discovery Using Etcd and Haproxy][5]


### Upgrade

The original [docker-register](https://github.com/jwilder/docker-register) only supports etcd v2.

It has been modified to support etcd v3. 

You can build the docker image with following command and follow the same usage.
```
docker build -t docker-register -f Dockerfile .
```


### Usage

To run it:

    $ docker run -d -e HOST_IP=1.2.3.4 -e ETCD_HOST=1.2.3.4:4001 -v /var/run/docker.sock:/var/run/docker.sock -t jwilder/docker-register

Then start any containers you want to be discoverable and publish their exposed port to the host.

    $ docker run -d -P -t ...

If you run the container on multiple hosts, they will be grouped together automatically.

### Limitations

There are a few simplications that were made:

* *Containers can only expose one port* - This is a simplification but if the container `EXPOSE`s
multiple ports, it won't be registered in etcd.
* *Exposed ports must be unique to the service* - Each container must expose it's service on a unique
port.  For example, if you have two different backend web services and they both expose their service
over port 80, then one will need to use a port 80 and the other a different port.


[1]: https://github.com/jwilder/docker-gen
[2]: http://jasonwilder.com/blog/2014/02/04/service-discovery-in-the-cloud/
[3]: http://nerds.airbnb.com/smartstack-service-discovery-cloud/
[4]: https://github.com/airbnb/nerve
[5]: http://jasonwilder.com/blog/2014/07/15/docker-service-discovery/

### TODO

* Support http, udp proxying
* Support multiple ports
* Make ETCD prefix configurable
* Support other backends (consul, zookeeper, redis, etc.)
