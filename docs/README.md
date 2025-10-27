# Setting up NRP Nautilus

### Get NRP Nautilus account
1. Point your browser to the [NRP Nautilus portal](http://nrp.ai/).
2. On the portal page, click on the **Log In** button at the top right corner.
3. You will be redirected to the **CILogon** page, where you have to **Select an Identity Provider**. Select your institution (for example: University Name) from the menu and click the **Log On** button.
4. After a successful authentication, you will log in to the portal and your account will be created if it was not created before.
5. Once you are added to a [namespace](https://nrp.ai/documentation/userdocs/start/using-nautilus/#namespace), your status will change to a cluster user and you will get access to all namespace resources.

### Sign up for Weights & Biases
1. Point your browser to the Weights & Biases homepage.
2. On the portal page, click on the **Sign Up** button at the top right corner.
3. Sign up with either channel and **Log In**.
4. On the bottom left, select **Create a team to collaborate** and create it.
5. On the top left, select **Create a new project** and create it.

### Set up NRP Nautilus access via kubectl
1. Install the Kubernetes command-line tool [kubectl](https://kubernetes.io/docs/tasks/tools/install-kubectl/) and the [kubelogin](https://github.com/int128/kubelogin?tab=readme-ov-file#setup) plugin.
2. Save this [config](https://nrp.ai/config) file as config (without any extension) and put it in your $HOME/.kube​ folder (If this folder does not exist on your machine, please create it beforehand).
3. Run kubectl config get-contexts in the command line to verify if you are using the correct config file. It should look like:
    ```
    $ kubectl config get-contexts
    CURRENT   NAME       CLUSTER    AUTHINFO   NAMESPACE
    *         nautilus   nautilus   oidc
    ```
4. Run `kubectl get nodes`​. It will authenticate via CiLogon by opening the browser window, and close one in the end.
5. Run `​kubectl get pods`. If you see the message “No resources found in your namespace” it means there are no pods in your namespace yet. This indicates that you have access to the resources of your namespace.

For more information, NRP Nautilus's official documentation is available at: [Getting Started with NRP Nautilus | NRP Nautilus](https://nrp.ai/documentation/userdocs/start/getting-started/).

# Mnist examples

### Create PVC
1. Create a file named `pvc.yaml`, replace `{YOUR_NAME}` with whatever you like:
    ```
    apiVersion: v1
    kind: PersistentVolumeClaim
    metadata:
    name: pvc-{YOUR_NAME}
    spec:
    affinity:
        nodeAffinity:
        requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
            - key: topology.kubernetes.io/region
                operator: In
                values:
                - us-west # assuming we are in CA
    storageClassName: rook-ceph-block # an RBD filesystem at west
    accessModes:
    - ReadWriteOnce
    resources:
        requests:
        storage: 1Gi
    ```
2. Run `kubectl apply -f pvc.yaml` in the command line.
3. Run `kubectl get pvc` to check the status of your pvc. `Bound` means success.

### Create Pod
1. Create a file named `pod.yaml`, replace `{YOUR_NAME}` with whatever you like:
    ```
    apiVersion: v1
    kind: Pod
    metadata:
    name: test-pod-{YOUR_NAME}
    spec:
    containers:
    - name: mypod
        image: ubuntu
        command: ["sh", "-c", "echo 'Im a new pod' && sleep infinity"]
        resources:
        limits:
            memory: 100Mi
            cpu: 100m
        requests:
            memory: 100Mi
            cpu: 100m
        volumeMounts: #optional, for mounting PVC
            - mountPath: /mnt/pvc
            name: storage-volume
    volumes:
        - name: storage-volume
        persistentVolumeClaim:
            claimName: pvc-{YOUR_NAME}

    ```
2. Run `kubectl create -f pod.yaml` in the command line.
3. Run `kubectl describe pod test-pod-{YOUR_NAME}` to get the status of the Pod.
4. Run `kubectl logs test-pod-{YOUR_NAME}` to check what is going on inside the Pod.
5. Run `kubectl exec -it test-pod-{YOUR_NAME} -- /bin/bash` to enter the Pod.
6. Run `echo "Hello world!" > /mnt/pvc/helloworld.txt` to write into your PVC.
7. Run `exit` to exit the Pod.
8. Run `kubectl delete pod test-pod-{YOUR_NAME}` to delete the Pod.

### Create Job
1. Create a file named `job.yaml`, replace `{YOUR_NAME}` with whatever you like:
    ```
    apiVersion: batch/v1
    kind: Job
    metadata:
    name: test-job-{YOUR_NAME}
    spec:
    template:
        spec:
        containers:
        - name: myjob
            image: perl
            command: ["perl",  "-Mbignum=bpi", "-wle", "print bpi(2000)"]
            resources:
            limits:
                memory: 200Mi
                cpu: 1
            requests:
                memory: 50Mi
                cpu: 50m
        restartPolicy: Never
    backoffLimit: 4
    ```
2. Run `kubectl create -f job.yaml` in the command line.
3. Run `kubectl describe job test-job-{YOUR_NAME}` to get the status of the Job.
4. Run `kubectl logs test-job-{YOUR_NAME}` to check what is going on inside the Job.
5. Run `kubectl delete job test-job-{YOUR_NAME}` to delete the Job.

# Warnings
### Don’t launch [interactive Jobs](https://nrp.ai/documentation/userdocs/start/policies/#_top)
The `sleep` command is not allowed in Job
### Don’t [force-delete Jobs/Pods](https://nrp.ai/documentation/userdocs/start/faq/#_top)
Do not use `kubectl delete --grace-period=0 --force` to delete stuck Pods! It will keep resources attached to the node for an indefinite period.
### Don’t [tolerate taints](https://nrp.ai/documentation/userdocs/running/special/#all-taints)
Please use caution when applying tolerations and only tolerate taints for which you have explicit authorization from cluster administrators.
### Read [cluster policies](https://nrp.ai/documentation/userdocs/start/policies) before use
The NRP Nautilus cluster has a documentation with all the DOs and DONTs. Please read before using the cluster to avoid getting banned.

### Get help in [the official forum](https://nrp.ai/contact)
Help, announcements, and updates are posted on the forum.