import React from 'react'
import mainImg from './images/mainImg.png';
import Sewing from './images/Sewing.png';
import Thread from './images/Thread.png';
import SewingThread from './images/SewingThread.png';
import { Link } from 'react-router-dom'

function NewProduct() {
  return (
    <div>
        <div>
            <header> 
                <button className="menu"> &#9776;</button>
                <h1> 烘焙工坊</h1>
                <nav>
                    <ul>
                        <li> <a href="/Home"> Home  </a></li>
                        <li> <a href="/NewProduct"> Product    </a></li>
                        <li> <a href="/Contactus"> Contact Us </a></li>
                        <li> <a href="/Login"> Login </a></li>
                    </ul>
                </nav>
            </header>  

            <main>
                <section className="productmain newsLayer">
                    <h2> Products </h2>
                    <p><h3> Provide customized services according to your needs. </h3></p>
                    <p> <Link to="/Contactus"> Learn More </Link></p>
                </section>

                <section className="slogan">
                    <blockquote> "Dyed, customized T-shirts with us, your gratification is our goal." </blockquote>
                </section>

                <section className="shop">
                    {/* <img src="./images/mainPageSide02.png" alt = "烘焙工坊大安店"/> */}
                    <img src = {mainImg} alt = "mainImg"/>
                    <div className="info">
                     <p>
                        <h2>  Product Introduction  </h2>
                        crafted with care and style in mind. From classic designs to trendy prints, 
                            each T-shirt is tailored to perfection to elevate your wardrobe. 
                            Embrace comfort and fashion effortlessly with our versatile range of T-shirts, 
                            designed to suit every occasion. </p>
        
                    </div>
                </section>

                <section className="product">
                    <div>
                        {/* <img src="./images/lacity03.png" alt = "Long-Sleeve T-Shirts"/> */}
                        <img src = {Sewing} alt = "sewing"/>
                        <h3>  Sewing </h3>
                        <p> Elevate your style with our expert stitching. 
                Every garment, from intricate designs to flawless finishes, 
                is crafted to perfection by our artisans. </p>
                    </div>
                    <div>
                        {/* <img src="./images/folddingTT.png" alt="Long-Sleeve T-Shirts "/> */}
                        <img src = {SewingThread} alt = "SewingThread"/>
                        <h3>  Sewing Thread  </h3>
                        <p> Enhance your fashion journey with our premium threads. 
                Carefully chosen for durability and strength,
                 they elevate the quality of your garments, stitch by stitch.</p>
                    </div>
                    <div>
                        {/* <img src="./images/hangingTgray.png" alt=" Unique Custom T-Shirts"/> */}
                        <img src = {Thread} alt = "Thread"/>
                        <h3>  Customize Color </h3>
                        <p>  Customized to be different from the crowd <br/> Allowing you to have a unique fashion experience </p>
                    </div>
                </section>

                <section className="newsletter">
                    <p>   Express yourself with our customizable colors. 
                From bold to subtle, our palette lets you stand out 
                and create a wardrobe that's uniquely you. </p>
                    <form>
                        <input type="email" placeholder="Please enter your Email address " required/>
                        <button type="submit"> Submit</button>
                    </form>
                </section>
            </main>

            <footer>
                <p> Copyright @ 2023 Fashion T-Shirt Manufacturing Company All Right Reserved </p>
            </footer>
        </div>
    </div>
  )
}

export default NewProduct