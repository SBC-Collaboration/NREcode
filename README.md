# NREcode
Nuclear recoil efficiency analysis code - for PICO study and in general

## Setting up your global git username and email
These are the identities that will be tied to any changes you make. These do not have to be your git username and email. Replace “Your name” with your own name and “Your email” with your own email address.

```
$ git config --global user.name "Your Name"
$ git config --global user.email "Your email"
```

## Cloning NREcode to your local directory
Navigate to where you want to clone NREcode. Note that cloning will create an "NREcode" folder.

```
$ git clone https://github.com/SBC-Collaboration/NREcode.git
```

## Setting remote origin
Before pushing to the master branch, you must first make sure you are a collaborator by asking the owner. Once you're a collaborator, run the following to make the master branch the default, replacing YOUR\_GITHUB\_USERNAME with your github.com username.

## Daily Workflow
* Start by pulling changes made by other users.

```
$ git pull origin master
```

* Make your own changes and stage them for comitting. If changing multiple files, it's generally a good idea to stage groups of files relating to the same change together.

```
$ git add darkmatter_locator.py
```

* Commit your changes. Comitting will record ALL of the files you've 'add'-ed to the repository. Note that this does NOT push your changes to the master branch

```
$ git commit -m "Added files that locate and display all the dark matter in the universe"
```

* Push your changes to the master branch. If you get an error about https request denied, make sure you complete the 'Setting remote origin' section.

```
$ git push
```
